# README

## Environment setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get Result

```sh
python3 result.py
```

Optionally set `CUDA_VISIBLE_DEVICES=<device id>` to limit the GPU usage.

This will load the model `btliu/llama3.2-1b-distilled` from Hugging Face Hub by default. If you want to test on the model trained in the [Prepare Model](#prepare-model) section, use the following command:

```sh
python3 result.py ./llama3.2-1b-distilled
```

## Prepare Model

Run the script to modify the `meta-llama/Llama-3.2-3B-Instruct` model for fast inference. This requires downloading Meta's llama model. Make sure you have access to it and set the `HF_TOKEN` environment variable to your hugging face token.

```sh
python3 main.py
```

Optionally set `CUDA_VISIBLE_DEVICES=<device id>` to limit the GPU usage.
