# README

## Environment setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare model

This requires downloading Meta's llama model. Make sure you have access to it and set the `HF_TOKEN` environment variable to your hugging face token.

```sh
python3 main.py
```

Optionally set `CUDA_VISIBLE_DEVICES=<device id>` to limit the GPU usage.

## Get Result

```sh
python3 result.py
```

Optionally set `CUDA_VISIBLE_DEVICES=<device id>` to limit the GPU usage.

