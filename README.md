---
language: en
license: mit
tags:
  - pretrained
  - causal-lm
  - fineweb-edu
  - custom-architecture
---

# tiny-edu-166m (ParchmentLM)

A 166M parameter transformer pretrained from scratch on 4B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

## Architecture (ParchmentLM)

Custom decoder-only transformer:
- **Parameters:** 166M
- **Layers:** 12
- **Hidden size:** 768
- **Attention heads:** 12
- **FFN:** SwiGLU (hidden=2048)
- **Context length:** 1024
- **Positional encoding:** RoPE (base=10000)
- **Normalization:** RMSNorm
- **Tokenizer:** cl100k_base (100277 tokens) — same as GPT-4

## Training

- **Dataset:** FineWeb-Edu 10BT sample
- **Tokens seen:** ~4B
- **Steps:** 30,000
- **Optimizer:** AdamW (lr=3e-4, cosine decay to 3e-5)
- **Hardware:** Single A100 80GB

## Installation

```bash
pip install transformers tiktoken
```

> **Note:** `tiktoken` is required because the tokenizer wraps OpenAI's cl100k_base encoding
> to guarantee byte-identical token IDs to the vocabulary the model was trained on.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("SlitherCode/tiny-edu-166m", trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained("SlitherCode/tiny-edu-166m", trust_remote_code=True)

inputs = tokenizer("The history of mathematics", return_tensors="pt")
out    = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.8, top_k=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## License

Model weights: MIT. Training data: ODC-By 1.0 attributing  -> fineweb
