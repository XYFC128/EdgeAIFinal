import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm


def prepare_tokenized_dataset(raw_dataset_split, tokenizer, max_seq_length, split_name="train"):
    """
    Prepares a dataset by concatenating, tokenizing, and chunking.
    Each example in the output dataset will have 'input_ids' and 'attention_mask'.
    """
    print(f"Preparing tokenized '{split_name}' dataset...")
    # 1. Concatenate all text samples
    # Filter out None or non-string entries first
    valid_texts = [text for text in raw_dataset_split["text"] if isinstance(text, str) and text.strip()]
    if not valid_texts:
        print(f"Warning: No valid text found in '{split_name}' split after filtering.")
        return Dataset.from_list([]) # Return an empty dataset

    concatenated_text = "\n\n".join(valid_texts)
    print(f"Total characters in concatenated '{split_name}' text: {len(concatenated_text)}")

    # 2. Tokenize the entire concatenated text
    # Llama tokenizer adds BOS by default if add_special_tokens is not False.
    # EOS is not added by default by Llama tokenizer. This matches typical pre-training.
    tokenized_output = tokenizer(
        concatenated_text,
        add_special_tokens=True, # Ensures BOS token is added at the beginning of the entire sequence
        return_attention_mask=False,
        return_tensors=None, # Get list of ids
        truncation=False,
    )
    all_token_ids = tokenized_output["input_ids"]
    print(f"Total tokens in '{split_name}' after tokenization: {len(all_token_ids)}")

    # 3. Chunk the token_ids into sequences of max_seq_length
    num_total_tokens = len(all_token_ids)
    # We drop the last partial chunk to ensure all sequences are max_seq_length
    num_samples = num_total_tokens // max_seq_length
    print(f"Number of full {max_seq_length}-token samples to be created for '{split_name}': {num_samples}")

    processed_examples = []
    if num_samples == 0:
        print(f"Warning: Not enough tokens ({num_total_tokens}) in '{split_name}' to create even one sample of length {max_seq_length}.")
        return Dataset.from_list([])


    for i in tqdm(range(num_samples), desc=f"Chunking '{split_name}' tokens"):
        start_index = i * max_seq_length
        end_index = (i + 1) * max_seq_length
        chunk_input_ids = all_token_ids[start_index:end_index]

        processed_examples.append({
            "input_ids": chunk_input_ids,
            "attention_mask": [1] * max_seq_length,
            # SFTTrainer with DataCollatorForLanguageModeling will create labels from input_ids
        })

    # 4. Create a Hugging Face Dataset
    hf_dataset = Dataset.from_list(processed_examples)
    print(f"Finished preparing tokenized '{split_name}' dataset with {len(hf_dataset)} samples.")
    return hf_dataset

def finetune_model(
    model_name,
    output_model_name,
    device="cuda",
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=None,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="paged_adamw_8bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=1e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_seq_length=1024,
    packing=False,
    use_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
    use_nested_quant=False,
    resume_from_checkpoint=None,
):
    """
    Fine-tune a model with LoRA using the specified configuration.
    """
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    
    print(f"Using device: {device}")

    # 2. Load Tokenizer and Model
    # ------------------------------------------------------------------------------------
    print(f"Loading base model: {model_name}")

    compute_dtype_torch = getattr(torch, bnb_4bit_compute_dtype)

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype_torch,
            bnb_4bit_use_double_quant=use_nested_quant,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype_torch if use_4bit else torch.float32,
        device_map=device,
    )
    model.config.use_cache = False # Required for gradient checkpointing
    model.config.pretraining_tp = 1 # Important for Llama models

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for causal LM

    # 3. Load and Prepare Dataset
    # ------------------------------------------------------------------------------------
    print(f"Loading raw dataset: {dataset_name} ({dataset_config_name})")
    raw_wiki_dataset = load_dataset(dataset_name, dataset_config_name)

    # Prepare the training dataset using the new method
    train_dataset_tokenized = prepare_tokenized_dataset(
        raw_wiki_dataset["train"],
        tokenizer,
        max_seq_length,
        split_name="train"
    )

    sft_eval_dataset = prepare_tokenized_dataset(
        raw_wiki_dataset["validation"],
        tokenizer,
        max_seq_length,
        split_name="validation"
    )

    print(f"Final tokenized training dataset size: {len(train_dataset_tokenized)}")
    print(f"Final tokenized validation dataset size: {len(sft_eval_dataset)}")

    # 4. PEFT Configuration
    # ------------------------------------------------------------------------------------
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Training Arguments
    # ------------------------------------------------------------------------------------
    output_dir = output_model_name + "-unmerged"
    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False, # Explicitly False. bf16 is controlled below.
        bf16=True if use_4bit and bnb_4bit_compute_dtype == "bfloat16" else False,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        group_by_length=False, # Data is already fixed length, no need to group
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",

        dataset_text_field=None,
        max_seq_length=max_seq_length,
        packing=packing,
    )

    # 6. Initialize SFTTrainer
    # ------------------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset_tokenized,
        eval_dataset=sft_eval_dataset,
        peft_config=peft_config,
    )

    # 7. Train the Model
    # ------------------------------------------------------------------------------------
    print("Starting training...")
    trainer.train(resume_from_checkpoint)

    # 8. Save the Fine-tuned Model (Adapter)
    # ------------------------------------------------------------------------------------
    print(f"Saving LoRA adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Free up VRAM
    del model
    del trainer
    if sft_eval_dataset:
        del sft_eval_dataset
    if train_dataset_tokenized:
        del train_dataset_tokenized
    torch.cuda.empty_cache()
    print("Training complete.")

    # 9. Merge LoRA adapter with base model and save
    # ------------------------------------------------------------------------------------
    print("\nAttempting to merge LoRA adapter with base model...")
    try:
        print(f"Loading base model ({model_name}) for merging in float16...")
        # Load in float16 for merging. bfloat16 could also work but fp16 is safer for wider compatibility.
        base_model_for_merging = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16, # Use float16 for merging
            device_map=device,
        )

        print(f"Loading PEFT adapter from {output_dir}...")
        merged_model = PeftModel.from_pretrained(base_model_for_merging, output_dir)

        print("Merging LoRA weights...")
        merged_model = merged_model.merge_and_unload()
        print("Merge complete.")

        print(f"Saving merged model to {output_model_name}...")
        merged_model.save_pretrained(output_model_name, safe_serialization=True)
        tokenizer.save_pretrained(output_model_name) # Tokenizer was already saved with adapter, but good practice
        print(f"Merged model saved to {output_model_name}")

        # Free up VRAM
        del base_model_for_merging
        del merged_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Could not merge and save model: {e}")

if __name__ == "__main__":
    finetune_model(
        model_name="./llama3-3.2-3b-pruned",
        output_model_name="llama3-3.2-3b-pruned-lora",
    )