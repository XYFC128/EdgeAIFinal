from finetune import finetune_model
from distillation import distill_model
from datasets import load_dataset
from ppl import model_ppl

if __name__ == "__main__":
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join(test_dataset["text"])

    finetune_model(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        output_model_name="./llama3.2-3b-wikitext2",
        num_train_epochs=2,
        learning_rate=1e-4,
        device_map="cuda:0",
        # resume_from_checkpoint=True,
    )

    ppl = model_ppl("./llama3.2-3b-wikitext2", eval_text)
    print(f"Perplexity of ./llama3.2-3b-wikitext2: {ppl}")

    distill_model(
        teacher_model_name="./llama3.2-3b-wikitext2",
        student_model_name="meta-llama/Llama-3.2-1B-Instruct",
        distillation_output_model_name="./llama3.2-1b-distilled",
        use_lora_on_student=True,
        student_lora_r=16,
        student_lora_alpha=32,
        num_train_epochs=3,
        eval_steps=20,
        device_map="cuda:0",
        # resume_from_checkpoint=True,
    )

    ppl = model_ppl("./llama3.2-1b-distilled", eval_text)
    print(f"Perplexity of ./llama3.2-1b-distilled: {ppl}")