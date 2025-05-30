from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    model_name = "./llama3.2-1b-distilled"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.name = "llama3.2-1b-distilled"
    model.push_to_hub("btliu/llama3.2-1b-distilled")
    tokenizer.push_to_hub("btliu/llama3.2-1b-distilled")