import os
import torch
import copy
import operator
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

from ppl import evaluate_ppl

class PassthroughLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position = None, # For Llama 3.1+
        **kwargs # Catch any other arguments
    ):
        """
        Accepts arguments a standard LlamaDecoderLayer might receive
        but only passes through hidden_states.
        Returns a tuple mimicking the output of a LlamaDecoderLayer.
        """
        attentions = None # This layer doesn't produce attentions

        # If use_cache is True, the model expects a present_key_value.
        # For a "removed" layer, it contributes no new KV state.
        # The model's forward loop will handle collecting KV states from *actual* layers.
        # We return None for this layer's contribution to the KV cache tuple.
        present_key_value_for_this_layer = None

        if use_cache:
            return hidden_states, present_key_value_for_this_layer, attentions
        else:
            return hidden_states, None, attentions


def prune_model(
    model_path: str,
    pruned_model_output_dir: str,
    k_blocks_to_prune=4,
    device="cuda",
    ppl_dataset_name="wikitext",
    ppl_dataset_config="wikitext-2-raw-v1",
    ppl_dataset_split="test",
    ppl_dataset_ratio=0.4
):
    """
    Prune a transformer model by removing the least important blocks.


    Args:
        model_path: Path to the model to be pruned
        pruned_model_output_dir: Directory to save the pruned model
        k_blocks_to_prune: Number of least important blocks to remove
        device: Device to use for computation ("cuda" or "cpu")
        ppl_dataset_name: Name of the dataset for PPL evaluation
        ppl_dataset_config: Configuration of the dataset
        ppl_dataset_split: Split of the dataset to use
        ppl_dataset_ratio: Ratio of dataset to use for evaluation
    """
    print(f"--- Structural Pruning of Transformer Blocks ---")
    print(f"Using device: {device}")

    # 1. Load Model and Tokenizer
    print(f"Loading model and tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load in full precision for pruning logic, quantization might interfere.
        # Use float16 for memory, bfloat16 if preferred and supported well by your PPL/model.
        original_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, # Helps with large models if loading to CPU first
            device_map=device if device == "cuda" else None # Load directly to GPU if cuda
        )
        if device == "cpu" and original_model.device.type != "cpu": # Ensure on CPU if device_map was None
            original_model.to(device)

    except Exception as e:
        print(f"ERROR: Failed to load model or tokenizer: {e}")
        return

    original_model.eval() # Set to evaluation mode

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ('{tokenizer.eos_token}')")

    # 2. Load data for PPL evaluation
    print(f"Loading PPL evaluation data: {ppl_dataset_name} ('{ppl_dataset_config}') split '{ppl_dataset_split}'")
    try:
        dataset = load_dataset(ppl_dataset_name, ppl_dataset_config, split=ppl_dataset_split)
        ppl_eval_texts = "\n\n".join(dataset["text"])
        ppl_eval_texts = ppl_eval_texts[:int(len(ppl_eval_texts) * ppl_dataset_ratio)]
        if not ppl_eval_texts:
            raise ValueError("No suitable texts found in PPL dataset after filtering or ppl_dataset_ratio is too low.")
        print(f"Using {len(ppl_eval_texts)} text samples for PPL evaluation.")
    except Exception as e:
        print(f"ERROR: Failed to load PPL dataset: {e}. Pruning cannot proceed without PPL data.")
        return

    # 3. Calculate Baseline PPL
    print("Calculating baseline PPL of the original model...")
    baseline_ppl = evaluate_ppl(original_model, tokenizer, ppl_eval_texts, device=device)
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    if math.isinf(baseline_ppl) or math.isnan(baseline_ppl) or baseline_ppl == 0:
        print("ERROR: Baseline PPL is invalid. Cannot proceed with pruning. Check PPL evaluation or model.")
        return

    # 4. Evaluate Importance Score of Each Block
    try:
        # For Llama-like models, layers are typically in model.model.layers
        if not (hasattr(original_model, 'model') and hasattr(original_model.model, 'layers')):
            raise AttributeError("Expected attribute 'model.layers' not found. Adjust for your model architecture.")
        transformer_blocks = original_model.model.layers
        num_original_blocks = len(transformer_blocks)
        print(f"Evaluating importance of {num_original_blocks} transformer blocks...")
    except AttributeError as e:
        print(f"ERROR: Could not access transformer blocks: {e}")
        return
        
    if k_blocks_to_prune >= num_original_blocks:
        print(f"Warning: k_blocks_to_prune ({k_blocks_to_prune}) is >= number of blocks ({num_original_blocks}).")
        print("This would remove all blocks, resulting in a non-functional model. Aborting.")
        return

    block_importances = [] # List of (block_index, ppl_increase, new_ppl_value)

    original_model.to("cpu") # Move original model to CPU to save GPU memory during evaluation

    for i in range(num_original_blocks):
        print(f"  Evaluating block {i+1}/{num_original_blocks} (index {i})...")
        
        # Create a deep copy to modify. This is memory-intensive.
        # Consider moving original_model to CPU if VRAM is tight for deepcopy and evaluation.
        if device == "cuda": torch.cuda.empty_cache()
        
        temp_model = copy.deepcopy(original_model) # This copies to CPU first
        temp_model.to(device) # Then move the copy to the target device

        # Replace the block with an Identity layer
        passthrough_layer = PassthroughLayer().to(device) # Ensure Identity is on the same device
        temp_model.model.layers[i] = passthrough_layer

        current_block_ppl = evaluate_ppl(temp_model, tokenizer, ppl_eval_texts, device=device)

        ppl_increase = current_block_ppl - baseline_ppl
        block_importances.append((i, ppl_increase, current_block_ppl))
        print(f"    Block {i}: PPL with block removed = {current_block_ppl:.4f}, PPL Increase = {ppl_increase:.4f}")

        del temp_model # Free memory from the temporary model
        if device == "cuda": torch.cuda.empty_cache()

    # Sort blocks by PPL increase (lower increase means less important)
    # A smaller (or more negative) ppl_increase means the block is less important.
    block_importances.sort(key=operator.itemgetter(1))

    print("\nBlock Importances (Sorted by PPL increase when removed - Lower values are less important):")
    for idx, increase, ppl_val in block_importances:
        print(f"  Block Index {idx}: PPL Increase = {increase:+.4f} (PPL became {ppl_val:.4f})")

    # 5. Prune the Model
    indices_to_remove_set = {imp[0] for imp in block_importances[:k_blocks_to_prune]}
    print(f"Pruning {k_blocks_to_prune} least important blocks with indices: {sorted(list(indices_to_remove_set))}")

    # Create the actual pruned model. We'll modify 'original_model' in-place for this.
    # Alternatively, one could create a new model from a modified config and copy weights.
    kept_layers = [
        layer for i, layer in enumerate(original_model.model.layers) # Use original_model.model.layers
        if i not in indices_to_remove_set
    ]

    if not kept_layers:
        print("ERROR: Pruning would remove all layers. This should have been caught. Aborting.")
        return

    # Directly modify the model's layer list and configuration
    original_model.model.layers = torch.nn.ModuleList(kept_layers)
    original_model.config.num_hidden_layers = len(kept_layers) # CRITICAL: Update config

    pruned_model = original_model # Renaming for clarity
    pruned_model.to(device)      # Ensure it's on the correct device

    print(f"Pruned model now has {len(pruned_model.model.layers)} transformer blocks.")

    # 6. Save the Pruned Model
    print(f"Saving pruned model to {pruned_model_output_dir}...")
    if not os.path.exists(pruned_model_output_dir):
        os.makedirs(pruned_model_output_dir)

    pruned_model.save_pretrained(pruned_model_output_dir)
    tokenizer.save_pretrained(pruned_model_output_dir)
    print(f"Pruned model saved successfully to {pruned_model_output_dir}.")

    # Free up VRAM
    del pruned_model
    if device == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    prune_model(
        model_path="./llama3-3.2-3b-wikitext2-lora-merged",
        pruned_model_output_dir="./llama3-3.2-3b-pruned",
        k_blocks_to_prune=6,
        device="cuda",
    )