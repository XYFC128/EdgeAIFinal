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

# 1. Configuration
# ------------------------------------------------------------------------------------
TEACHER_MODEL_NAME = "./llama-wikitext"
STUDENT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "wikitext"
DATASET_CONFIG_NAME = "wikitext-2-raw-v1"
DISTILLATION_OUTPUT_DIR = "./llama3-1b-distilled-wikitext2"

# Distillation Hyperparameters
TEMPERATURE = 2.0  # Temperature for softening probabilities
ALPHA_CE = 0.5     # Weight for student's own cross-entropy loss (next token prediction)
ALPHA_KD = 0.5     # Weight for KL divergence distillation loss (matching teacher)

# Student Model Training/LoRA Configuration
USE_LORA_ON_STUDENT = True # Set to True to use LoRA on the student
STUDENT_LORA_R = 16
STUDENT_LORA_ALPHA = 32
STUDENT_LORA_DROPOUT = 0.05
STUDENT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# TrainingArguments for Student
NUM_TRAIN_EPOCHS = 3 # Distillation might take more epochs
PER_DEVICE_BATCH_SIZE = 1 # Start very low due to two models
GRADIENT_ACCUMULATION_STEPS = 16 # Effective batch size = 1*16 = 16
LEARNING_RATE = 5e-5 # Typical learning rate for student
OPTIM = "adamw_torch" # Standard optimizer
MAX_SEQ_LENGTH = 1024

# Quantization for Teacher (to save VRAM)
LOAD_TEACHER_IN_4BIT = True
BNB_4BIT_COMPUTE_DTYPE_TEACHER = "bfloat16" # or "float16" if bfloat16 causes issues on T4
BNB_4BIT_QUANT_TYPE_TEACHER = "nf4"

# Quantization for Student (Optional, if full student model is too large for VRAM even for 1B)
LOAD_STUDENT_IN_4BIT = False # Usually, student is not quantized during training unless necessary
BNB_4BIT_COMPUTE_DTYPE_STUDENT = "bfloat16"
BNB_4BIT_QUANT_TYPE_STUDENT = "nf4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: Distillation on CPU will be extremely slow. GPU is highly recommended.")
    LOAD_TEACHER_IN_4BIT = False
    LOAD_STUDENT_IN_4BIT = False

# 2. Load Tokenizer (assuming same tokenizer for teacher and student from Llama family)
# ------------------------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Load Models
# ------------------------------------------------------------------------------------
print(f"Loading teacher model: {TEACHER_MODEL_NAME}")
teacher_bnb_config = None
if LOAD_TEACHER_IN_4BIT:
    teacher_compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE_TEACHER)
    teacher_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE_TEACHER,
        bnb_4bit_compute_dtype=teacher_compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_NAME, # Or TEACHER_MODEL_NAME_FINETUNED if you use a fine-tuned teacher
    quantization_config=teacher_bnb_config,
    torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE_TEACHER) if LOAD_TEACHER_IN_4BIT else torch.float16,
    device_map="auto",
)
teacher_model.eval() # Set teacher to evaluation mode
for param in teacher_model.parameters(): # Freeze teacher
    param.requires_grad = False
print("Teacher model loaded and frozen.")

print(f"Loading student model: {STUDENT_MODEL_NAME}")
student_bnb_config = None
if LOAD_STUDENT_IN_4BIT: # Usually false for student during training
    student_compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE_STUDENT)
    student_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE_STUDENT,
        bnb_4bit_compute_dtype=student_compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_NAME,
    quantization_config=student_bnb_config,
    torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE_STUDENT) if LOAD_STUDENT_IN_4BIT else torch.float16, # Or bfloat16
    device_map="auto", # This might place parts on CPU if not enough VRAM for student
)
student_model.train() # Set student to training mode

if USE_LORA_ON_STUDENT:
    print("Applying LoRA to student model...")
    student_peft_config = LoraConfig(
        r=STUDENT_LORA_R,
        lora_alpha=STUDENT_LORA_ALPHA,
        lora_dropout=STUDENT_LORA_DROPOUT,
        target_modules=STUDENT_LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    student_model = get_peft_model(student_model, student_peft_config)
    student_model.print_trainable_parameters()
print("Student model loaded.")

# 4. Load and Prepare Dataset
# ------------------------------------------------------------------------------------
print(f"Loading and preparing dataset: {DATASET_NAME} ({DATASET_CONFIG_NAME})")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG_NAME)
# Process train and validation sets
train_dataset = prepare_tokenized_dataset(dataset["train"], tokenizer, max_seq_length=MAX_SEQ_LENGTH, split_name="train")
eval_dataset = prepare_tokenized_dataset(dataset["validation"], tokenizer, max_seq_length=MAX_SEQ_LENGTH, split_name="validation")

print(f"Train dataset size after processing: {len(train_dataset)}")
print(f"Eval dataset size after processing: {len(eval_dataset)}")
if len(train_dataset) > 0:
    print(f"First training example input_ids: {train_dataset[0]['input_ids'][:10]}...")
else:
    print("Training dataset is empty. Check preprocessing or dataset source.")
    exit()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Custom Distillation Trainer
# ------------------------------------------------------------------------------------
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

# 6. Training Arguments for Student
# ------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=DISTILLATION_OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    optim=OPTIM,
    # For T4 with bfloat16 compute_dtype in BNB, bf16=True might be an option
    # However, student is loaded in fp16 by default here. So fp16=True makes sense if not using BNB for student.
    # If student is BNB with bfloat16 compute, then bf16=True
    fp16=True if not LOAD_STUDENT_IN_4BIT and student_model.dtype == torch.float16 else False,
    bf16=True if LOAD_STUDENT_IN_4BIT and BNB_4BIT_COMPUTE_DTYPE_STUDENT == "bfloat16" else False,
    logging_dir=f"{DISTILLATION_OUTPUT_DIR}/logs",
    logging_steps=10,
    save_steps=20,
    eval_steps=20,
    eval_strategy="steps", # Evaluate at the end of each epoch
    load_best_model_at_end=True, # Load the best model based on eval loss
    metric_for_best_model="eval_loss", # Default is eval_loss
    greater_is_better=False,
    report_to="none",
    # gradient_checkpointing=True, # Can save memory if student is not LoRA and large
    remove_unused_columns=False, # Important for custom compute_loss
)

# 7. Initialize Trainer and Start Distillation
# ------------------------------------------------------------------------------------
distiller = DistillationTrainer(
    model=student_model, # Student model
    teacher_model=teacher_model,
    temperature=TEMPERATURE,
    alpha_ce=ALPHA_CE,
    alpha_kd=ALPHA_KD,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting distillation training...")
distiller.train(resume_from_checkpoint=True)

# 8. Save the Distilled Student Model
# ------------------------------------------------------------------------------------
print(f"Saving distilled student model to {DISTILLATION_OUTPUT_DIR}")
distiller.save_model(DISTILLATION_OUTPUT_DIR) # Saves the student model
# If LoRA was used on student, this saves the adapter.
# To save a merged LoRA student model, you'd follow similar steps as before.

tokenizer.save_pretrained(DISTILLATION_OUTPUT_DIR)
print("Distillation complete. Student model and tokenizer saved.")

# To merge if LoRA was used on student:
if USE_LORA_ON_STUDENT:
    print("Merging LoRA weights for student model...")
    base_student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    merged_student_model = PeftModel.from_pretrained(base_student_model, DISTILLATION_OUTPUT_DIR)
    merged_student_model = merged_student_model.merge_and_unload()
    merged_student_model.save_pretrained(f"{DISTILLATION_OUTPUT_DIR}-merged")
    tokenizer.save_pretrained(f"{DISTILLATION_OUTPUT_DIR}-merged")
    print("Merged student model saved.")