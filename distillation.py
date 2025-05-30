import os
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from finetune import prepare_tokenized_dataset

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model, temperature, alpha_ce, alpha_kd, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        # Ensure teacher is on the same device as student and data
        if self.args.n_gpu > 0 : # or self.args.local_rank != -1 for DDP
             self.teacher_model.to(self.args.device)


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # `model` is the student_model
        # `inputs` already contains 'labels' thanks to DataCollatorForLanguageModeling
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        loss_ce = student_outputs.loss # This is the student's own cross-entropy loss

        # Get teacher logits (no gradients needed for teacher)
        with torch.no_grad():
            # Ensure inputs for teacher don't include labels if teacher model complains
            teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

        # Apply temperature scaling and compute KL divergence
        active_loss = inputs['labels'].view(-1) != -100 # Mask for non-padding tokens
        
        active_student_logits = student_logits.view(-1, student_logits.size(-1))[active_loss]
        active_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))[active_loss]

        if active_student_logits.numel() == 0 or active_teacher_logits.numel() == 0: # No active tokens in batch
            # Handle cases where all tokens are masked, or if teacher/student logits somehow became empty
            # This can happen if input sequences are very short and fully masked after truncation/tokenization
            loss_kd = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
            if loss_ce is None: # If student_outputs.loss was also None (e.g. no labels)
                 loss_ce = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)

        else:
            soft_student_log_probs = F.log_softmax(active_student_logits / self.temperature, dim=-1)
            soft_teacher_probs = F.softmax(active_teacher_logits / self.temperature, dim=-1)
            
            loss_kd = F.kl_div(soft_student_log_probs, soft_teacher_probs, reduction='batchmean', log_target=False) * (self.temperature ** 2)

        # Ensure loss_ce is a valid tensor if it was None (can happen if all labels are -100 in a batch)
        if loss_ce is None:
            loss_ce = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)


        total_loss = self.alpha_ce * loss_ce + self.alpha_kd * loss_kd


        if return_outputs:
            # Construct a compatible output dictionary
            # The base Trainer expects loss as the first element and then a dict of outputs
            # student_outputs already has 'logits', 'loss' (original CE loss), etc.
            # We add our custom losses to this dict for potential logging or inspection.
            output_dict = {"loss_ce": loss_ce, "loss_kd": loss_kd}
            if hasattr(student_outputs, "logits"):
                output_dict["logits"] = student_outputs.logits
            if hasattr(student_outputs, "hidden_states"):
                output_dict["hidden_states"] = student_outputs.hidden_states
            if hasattr(student_outputs, "attentions"):
                output_dict["attentions"] = student_outputs.attentions
            return (total_loss, output_dict)
        
        return total_loss


def distill_model(
    teacher_model_name="./llama-wikitext",
    student_model_name="meta-llama/Llama-3.2-1B-Instruct",
    distillation_output_model_name="./llama3-1b-distilled-wikitext2",
    device="cuda",
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    temperature=2.0,
    alpha_ce=0.5,
    alpha_kd=0.5,
    use_lora_on_student=True,
    student_lora_r=16,
    student_lora_alpha=32,
    student_lora_dropout=0.05,
    student_lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    num_train_epochs=3,
    eval_steps=20,
    per_device_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    optim="adamw_torch",
    max_seq_length=1024,
    load_teacher_in_4bit=True,
    bnb_4bit_compute_dtype_teacher="bfloat16",
    bnb_4bit_quant_type_teacher="nf4",
    load_student_in_4bit=False,
    bnb_4bit_compute_dtype_student="bfloat16",
    bnb_4bit_quant_type_student="nf4",
    resume_from_checkpoint=None,
):
    """
    Run knowledge distillation from a teacher model to a student model.
    
    Args:
        teacher_model_name: Path or name of the teacher model
        student_model_name: Path or name of the student model
        distillation_output_model_name: Directory to save the distilled model
        device: Device to use for training ('cuda' or 'cpu')
        dataset_name: Name of the dataset to use
        dataset_config_name: Configuration name for the dataset
        temperature: Temperature for softening probabilities in distillation
        alpha_ce: Weight for student's cross-entropy loss
        alpha_kd: Weight for KL divergence distillation loss
        use_lora_on_student: Whether to use LoRA on student model
        student_lora_r: LoRA rank parameter
        student_lora_alpha: LoRA alpha parameter
        student_lora_dropout: LoRA dropout rate
        student_lora_target_modules: Target modules for LoRA
        num_train_epochs: Number of training epochs
        eval_steps: Number of steps between evaluations
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate for training
        optim: Optimizer to use
        max_seq_length: Maximum sequence length for tokenization
        load_teacher_in_4bit: Whether to load teacher model in 4-bit quantization
        bnb_4bit_compute_dtype_teacher: Compute dtype for teacher quantization
        bnb_4bit_quant_type_teacher: Quantization type for teacher
        load_student_in_4bit: Whether to load student model in 4-bit quantization
        bnb_4bit_compute_dtype_student: Compute dtype for student quantization
        bnb_4bit_quant_type_student: Quantization type for student
        resume_from_checkpoint: Set to true to resume training from a checkpoint or specify a checkpoint name
    """
    print(f"Using device: {device} for distillation")
    if device == "cpu":
        print("WARNING: Distillation on CPU will be extremely slow. GPU is highly recommended.")
        load_teacher_in_4bit = False
        load_student_in_4bit = False

    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Teacher Model
    print(f"Loading teacher model: {teacher_model_name}")
    teacher_bnb_config = None
    if load_teacher_in_4bit:
        teacher_compute_dtype = getattr(torch, bnb_4bit_compute_dtype_teacher)
        teacher_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type_teacher,
            bnb_4bit_compute_dtype=teacher_compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        quantization_config=teacher_bnb_config,
        torch_dtype=getattr(torch, bnb_4bit_compute_dtype_teacher) if load_teacher_in_4bit else torch.float16,
        device_map="auto",
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print("Teacher model loaded and frozen.")

    # Load Student Model
    print(f"Loading student model: {student_model_name}")
    student_bnb_config = None
    if load_student_in_4bit:
        student_compute_dtype = getattr(torch, bnb_4bit_compute_dtype_student)
        student_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type_student,
            bnb_4bit_compute_dtype=student_compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        quantization_config=student_bnb_config,
        torch_dtype=getattr(torch, bnb_4bit_compute_dtype_student) if load_student_in_4bit else torch.float16,
        device_map="auto",
    )
    student_model.train()

    if use_lora_on_student:
        print("Applying LoRA to student model...")
        student_peft_config = LoraConfig(
            r=student_lora_r,
            lora_alpha=student_lora_alpha,
            lora_dropout=student_lora_dropout,
            target_modules=student_lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        student_model = get_peft_model(student_model, student_peft_config)
        student_model.print_trainable_parameters()
    print("Student model loaded.")

    # Load and Prepare Dataset
    print(f"Loading and preparing dataset: {dataset_name} ({dataset_config_name})")
    dataset = load_dataset(dataset_name, dataset_config_name)
    train_dataset = prepare_tokenized_dataset(dataset["train"], tokenizer, max_seq_length=max_seq_length, split_name="train")
    eval_dataset = prepare_tokenized_dataset(dataset["validation"], tokenizer, max_seq_length=max_seq_length, split_name="validation")

    print(f"Train dataset size after processing: {len(train_dataset)}")
    print(f"Eval dataset size after processing: {len(eval_dataset)}")
    if len(train_dataset) > 0:
        print(f"First training example input_ids: {train_dataset[0]['input_ids'][:10]}...")
    else:
        print("Training dataset is empty. Check preprocessing or dataset source.")
        return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    distillation_output_dir = distillation_output_model_name if use_lora_on_student else distillation_output_model_name + "-unmerged"
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=distillation_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        optim=optim,
        fp16=True if not load_student_in_4bit and student_model.dtype == torch.float16 else False,
        bf16=True if load_student_in_4bit and bnb_4bit_compute_dtype_student == "bfloat16" else False,
        logging_dir=f"{distillation_output_dir}/logs",
        logging_steps=10,
        save_steps=eval_steps, # save_step must equal to eval_steps for load_best_model_at_end to work
        eval_steps=eval_steps,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )

    # Initialize Trainer and Start Distillation
    distiller = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        temperature=temperature,
        alpha_ce=alpha_ce,
        alpha_kd=alpha_kd,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting distillation training...")
    distiller.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the Distilled Student Model
    print(f"Saving distilled student model to {distillation_output_dir}")
    distiller.save_model(distillation_output_dir)
    tokenizer.save_pretrained(distillation_output_dir)
    print("Distillation complete. Student model and tokenizer saved.")

    # Clear VRAM
    del teacher_model
    del student_model
    del distiller
    torch.cuda.empty_cache()

    # Merge LoRA if used
    if use_lora_on_student:
        print("Merging LoRA weights for student model...")
        base_student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        merged_student_model = PeftModel.from_pretrained(base_student_model, distillation_output_dir)
        merged_student_model = merged_student_model.merge_and_unload()
        merged_student_model.save_pretrained(distillation_output_model_name)
        tokenizer.save_pretrained(distillation_output_model_name)
        print("Merged student model saved.")

        # Clear VRAM
        del merged_student_model
        del base_student_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    distill_model(
        teacher_model_name="./llama-wikitext",
        student_model_name="meta-llama/Llama-3.2-1B-Instruct",
        distillation_output_model_name="./distill-test",
    )