import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
from tqdm import tqdm # For progress bar

def load_model_and_tokenizer(model_path: str):
    print(f"Loading model {model_path}")
    # For merged model, load in higher precision if possible (e.g., float16)
    # T4 should handle 3B in float16 for inference
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, # Or bfloat16 if preferred and supported well
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval() # Set to evaluation mode
    return model, tokenizer


def evaluate_ppl(model, tokenizer, eval_text, seq_len=2048, device="cuda:0"):
    test_enc = tokenizer(eval_text, return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // seq_len
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating PPL..."):
        batch = test_enc[:, (i * seq_len):((i + 1) * seq_len)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * seq_len):((i + 1) * seq_len)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seq_len
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    
    return ppl.item()


def model_ppl(model_path: str, eval_text: str, seq_len=2048, device="cuda:0"):
    model, tokenizer = load_model_and_tokenizer(model_path)
    ppl = evaluate_ppl(model, tokenizer, eval_text, seq_len, device)
    del model
    del tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()  # Free up GPU memory
    return ppl


if __name__ == "__main__":
    
    # --- Configuration ---
    # MODEL_PATH = "./llama3-3.2-3b-wikitext2-lora-merged" # Path to your merged model
    MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")
    model = MODEL_PATH
    if len(sys.argv) > 1:
        model = sys.argv[1]
    # 1. Load Model and Tokenizer
    model_ppl, tokenizer_ppl = load_model_and_tokenizer(model)

    # 2. Load Evaluation Dataset
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join(test_dataset["text"])

    # 3. Calculate Perplexity
    print("Calculating perplexity...")
    perplexity_score = evaluate_ppl(model_ppl, tokenizer_ppl, eval_text, device=DEVICE)

    print(f"PPL: {perplexity_score:.4f}")

    # Clean up (optional, good practice if running multiple things)
    del model_ppl
    del tokenizer_ppl
    if DEVICE == "cuda":
        torch.cuda.empty_cache()