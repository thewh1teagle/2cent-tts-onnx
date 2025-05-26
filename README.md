# 2cent-tts-onnx

Based on [2cent-tts](https://github.com/taylorchu/2cent-tts)

## Setup

```console
uv sync
wget https://github.com/taylorchu/2cent-tts/releases/download/v0.2.0/2cent.gguf
wget https://github.com/taylorchu/2cent-tts/releases/download/v0.2.0/tokenizer.json

uv run src/main.py
```