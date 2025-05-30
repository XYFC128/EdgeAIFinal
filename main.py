from finetune import finetune_model
from prune import prune_model
from datasets import load_dataset
from ppl import model_ppl

if __name__ == "__main__":
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join(test_dataset["text"])

    finetune_model(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        output_dir="./llama-wikitext-unmerged",
        merged_model_name="./llama-wikitext",
        num_train_epochs=2,
        learning_rate=1e-4,
        device='cuda:0',
        resume_from_checkpoint=True,
    )

    ppl = model_ppl("./llama-wikitext", eval_text)
    print(f"Perplexity of ./llama-wikitext: {ppl}")

    # prune_model(
    #     model_path="./llama-wikitext",
    #     pruned_model_output_dir="./llama-pruned",
    #     k_blocks_to_prune=4,
    #     device='cuda'
    # )

    # ppl = model_ppl("./llama-pruned", eval_text)
    # print(f"Perplexity of ./llama-pruned: {ppl}")

    # finetune_model(
    #     model_name="./llama-pruned",
    #     output_dir="./llama-pruned-lora-unmerged",
    #     merged_model_name="./llama-pruned-lora",
    #     num_train_epochs=2,
    #     learning_rate=1e-4,
    #     device='cuda:0',
    #     resume_from_checkpoint=True,
    # )

    # ppl = model_ppl("./llama-pruned-lora", eval_text)
    # print(f"Perplexity of ./llama-pruned-lora: {ppl}")

    # prune_model(
    #     model_path="./llama-pruned-lora",
    #     pruned_model_output_dir="./llama-pruned2",
    #     k_blocks_to_prune=1,
    #     device='cuda'
    # )

    # ppl = model_ppl("./llama-pruned2", eval_text)
    # print(f"Perplexity of ./llama-pruned2: {ppl}")

    # finetune_model(
    #     model_name="./llama-pruned2",
    #     output_dir="./llama-pruned2-lora-unmerged",
    #     merged_model_name="./llama-pruned2-lora",
    #     num_train_epochs=2,
    #     learning_rate=1e-4,
    #     device='cuda:0',
    #     # resume_from_checkpoint=True,
    # )

    # ppl = model_ppl("./llama-pruned2-lora", eval_text)
    # print(f"Perplexity of ./llama-pruned2-lora: {ppl}")

    # prune_model(
    #     model_path="./llama3-1b-distilled-wikitext2-merged",
    #     pruned_model_output_dir="./llama3-1b-distilled-wikitext2-pruned",
    #     k_blocks_to_prune=1,
    #     device='cuda'
    # )

    # ppl = model_ppl("./llama3-1b-distilled-wikitext2-pruned", eval_text)
    # print(f"Perplexity of ./llama3-1b-distilled-wikitext2-pruned: {ppl}")

    # finetune_model(
    #     model_name="./llama3-1b-distilled-wikitext2-pruned",
    #     output_dir="./llama3-1b-distilled-wikitext2-pruned-unmerged",
    #     merged_model_name="./llama3-1b-distilled-wikitext2-pruned-lora",
    #     num_train_epochs=2,
    #     learning_rate=1e-4,
    #     lora_r=32,
    #     lora_alpha=64,
    #     device='cuda:0',
    #     # resume_from_checkpoint=True,
    # )

    # ppl = model_ppl("./llama3-1b-distilled-wikitext2-pruned-lora", eval_text)
    # print(f"Perplexity of ./llama3-1b-distilled-wikitext2-pruned-lora: {ppl}")