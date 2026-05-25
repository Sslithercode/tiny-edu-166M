"""
Export tiny-edu-166M-instruct-v3 to ONNX and bundle the tokenizer.

Usage:
    python export_onnx.py

Outputs (in ./onnx_bundle/):
    parchment_instruct.onnx   - the model graph (~650 MB fp32)
    tokenizer/                - saved tokenizer files (for save_pretrained reload)
    tiktoken_cache/           - offline tiktoken .tiktoken data file
    infer_onnx.py             - self-contained greedy inference demo

Key challenges solved here:
  - RoPE lazy cache: _cos_cache/_sin_cache are Python attributes (not buffers),
    built on first forward.  We warm them up before tracing so the tracer sees
    the "cache already populated" branch and bakes the full 1024-row constant
    into the graph; the [:, :, :seq] slice remains dynamic.
  - Clean ONNX signature: wrap the model so forward is just
    input_ids (LongTensor) -> logits (FloatTensor).
  - Tokenizer bundling: copy the tiktoken cache file so inference can run
    fully offline once the bundle is distributed.
"""

import os
import sys
import io
import shutil
import json
from pathlib import Path

# Windows CP1252 terminals can't encode emoji that PyTorch prints internally
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH    = "./tiny-edu-166M-instruct-v3"
TOKENIZER_HF  = "SlitherCode/tiny-edu-166m"   # remote tokenizer source
OUTPUT_DIR    = Path("./onnx_bundle")
OUTPUT_DIR.mkdir(exist_ok=True)
ONNX_PATH     = OUTPUT_DIR / "parchment_instruct.onnx"


# ── 1. load model ──────────────────────────────────────────────────────────────
print("Loading model from", MODEL_PATH)
# config.json says dtype=bfloat16; force float32 because onnxruntime's CPU
# provider has no Pow(bfloat16) kernel — everything must be fp32 for export.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, dtype=torch.float32
)
model.eval()


# ── 2. warm up RoPE lazy caches ────────────────────────────────────────────────
# RoPE._cos_cache / ._sin_cache start as None and are built on first forward.
# Running a dummy pass here ensures they're populated (and set to the right
# device/dtype) before we hand the module to the ONNX tracer.
# The tracer then sees the "cache is already a tensor" branch and emits the
# cos/sin tables as fp32 constants in the graph.
# The [:, :, :seq] slice still uses a dynamic 'seq' axis, so variable-length
# inputs work correctly at runtime.
print("Warming up RoPE caches ...")
_dummy_warmup = torch.zeros(1, 16, dtype=torch.long)
with torch.no_grad():
    _ = model(_dummy_warmup)


# ── 3. clean wrapper for ONNX ──────────────────────────────────────────────────
# The original forward() returns CausalLMOutputWithPast and accepts **kwargs.
# ONNX needs a plain tensor -> tensor signature.
class _OnnxWrapper(torch.nn.Module):
    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # attention_mask is ignored by ParchmentModel (uses is_causal=True)
        return self.m(input_ids=input_ids).logits

wrapper = _OnnxWrapper(model).eval()


# ── 4. export ──────────────────────────────────────────────────────────────────
print(f"Exporting to {ONNX_PATH} (this may take a minute) ...")
example_input = torch.zeros(1, 16, dtype=torch.long)

# RoPE's _cos_cache/_sin_cache are plain Python attrs populated by warmup.
# The legacy JIT tracer (torch.onnx.export) evaluates the `if None` check
# at trace time (takes the False branch) and bakes the tensors as constants.
# No buffer registration needed — that would conflict with the existing attr.
torch.onnx.export(
    wrapper,
    (example_input,),
    str(ONNX_PATH),
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "logits":    {0: "batch", 1: "seq"},
    },
    opset_version=17,
    do_constant_folding=True,
    # Force the legacy JIT tracer (not the dynamo exporter).
    # The dynamo exporter requires dynamic_shapes instead of dynamic_axes,
    # and may not handle the RoPE Python-attribute caches as constants.
    dynamo=False,
)
size_mb = ONNX_PATH.stat().st_size / 1e6
print(f"  done — {size_mb:.1f} MB")


# ── 5. save tokenizer locally ──────────────────────────────────────────────────
# save_pretrained writes tokenizer_config.json + tiktoken_encoding.json.
# The tiktoken_encoding.json only contains {"encoding_name": "cl100k_base"},
# so tiktoken still needs to download the actual vocab file once.
# Step 6 copies that file so subsequent loads can run offline.
print("Saving tokenizer to", OUTPUT_DIR / "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_HF, trust_remote_code=True)
tokenizer.save_pretrained(str(OUTPUT_DIR / "tokenizer"))


# ── 6. copy tiktoken cache for offline use ────────────────────────────────────
# tiktoken downloads the cl100k_base vocab to ~/.cache/tiktoken/ (or
# $TIKTOKEN_CACHE_DIR).  We copy those files into the bundle so the inference
# script can point TIKTOKEN_CACHE_DIR at the bundle and never phone home.
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")   # triggers download if not cached

cache_env = os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR")
if cache_env:
    tik_src = Path(cache_env)
else:
    import tempfile
    tik_src = Path(tempfile.gettempdir()) / "data-gym-cache"
tik_dst   = OUTPUT_DIR / "tiktoken_cache"
tik_dst.mkdir(exist_ok=True)

copied = 0
if tik_src.exists():
    for f in tik_src.iterdir():
        shutil.copy2(f, tik_dst / f.name)
        copied += 1
    print(f"  copied {copied} tiktoken cache file(s) → {tik_dst}")
else:
    print(f"  WARNING: tiktoken cache not found at {tik_src}")
    print("  Users will need internet access on first run to download cl100k_base.")

# also save a small metadata file the inference script can read
(OUTPUT_DIR / "bundle_meta.json").write_text(json.dumps({
    "model_file":       "parchment_instruct.onnx",
    "tokenizer_dir":    "tokenizer",
    "tiktoken_cache":   "tiktoken_cache",
    "encoding_name":    "cl100k_base",
    "vocab_size":       enc.n_vocab,
    "eot_token_id":     enc.eot_token,
    # special tokens the instruct model uses
    "special_tokens": {
        name: tok_id
        for name, tok_id in enc._special_tokens.items()
    },
    "chat_template_note": (
        "apply_chat_template is on the HF tokenizer; "
        "see infer_onnx.py for a manual implementation."
    ),
}, indent=2))


# ── 7. write a self-contained inference script ────────────────────────────────
infer_script = r"""
\"\"\"
Offline inference with the exported ONNX model.
Requires: pip install onnxruntime tiktoken
\"\"\"
import os, sys, json
from pathlib import Path

BUNDLE = Path(__file__).parent

# Point tiktoken at the bundled cache so it never hits the network.
# tiktoken checks TIKTOKEN_CACHE_DIR first, then DATA_GYM_CACHE_DIR,
# then tempdir/data-gym-cache — set both to be safe.
os.environ["TIKTOKEN_CACHE_DIR"] = str(BUNDLE / "tiktoken_cache")
os.environ["DATA_GYM_CACHE_DIR"] = str(BUNDLE / "tiktoken_cache")

import tiktoken
import numpy as np
import onnxruntime as ort

meta      = json.loads((BUNDLE / "bundle_meta.json").read_text())
enc       = tiktoken.get_encoding(meta["encoding_name"])
EOT_ID    = meta["eot_token_id"]
# special token ids used by the instruct chat template
SPL       = meta["special_tokens"]

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
available = ort.get_available_providers()
providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
sess = ort.InferenceSession(
    str(BUNDLE / meta["model_file"]),
    sess_options=sess_opts,
    providers=providers,
)

MAX_SEQ   = 1024
END_IDS   = {EOT_ID, SPL.get("<|endoftext|>", EOT_ID)}


def _apply_chat_template(system: str, user: str) -> list[int]:
    \"\"\"
    Manual chat template matching the training format (chat_template.jinja):
      {role}\\n{content}<|endoftext|>\\n   (for system and user turns)
      assistant\\n                          (generation prompt)
    \"\"\"
    def role_block(role: str, content: str) -> list[int]:
        return enc.encode(role + "\n" + content, allowed_special="all") + [EOT_ID] + enc.encode("\n", allowed_special="all")

    ids  = role_block("system", system)
    ids += role_block("user", user)
    ids += enc.encode("assistant\n", allowed_special="all")
    return ids


def generate(
    prompt_ids: list[int],
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_k: int = 50,
) -> str:
    tokens = list(prompt_ids)
    prompt_len = len(tokens)

    for _ in range(max_new_tokens):
        window   = tokens[-MAX_SEQ:]
        inp      = np.array([window], dtype=np.int64)
        (logits,) = sess.run(["logits"], {"input_ids": inp})
        last      = logits[0, -1, :].astype(np.float32)

        last /= max(temperature, 1e-8)
        if top_k:
            kth = np.partition(last, -top_k)[-top_k]
            last[last < kth] = -1e9

        probs = np.exp(last - last.max())
        probs /= probs.sum()
        next_id = int(np.random.choice(len(probs), p=probs))

        tokens.append(next_id)
        if next_id in END_IDS:
            break

    new_ids = tokens[prompt_len:]
    # strip trailing special tokens
    while new_ids and new_ids[-1] in END_IDS:
        new_ids.pop()
    return enc.decode(new_ids)


def ask(question: str, system: str = "You are a helpful assistant.") -> str:
    ids = _apply_chat_template(system, question)
    return generate(ids)


if __name__ == "__main__":
    questions = [
        "1 + 2 =",
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask(q)}")
        print()
"""

# strip the escaped quotes we needed inside the f-string
infer_script = infer_script.replace(r'\"\"\"', '"""')
(OUTPUT_DIR / "infer_onnx.py").write_text(infer_script.lstrip("\n"), encoding="utf-8")
print(f"Wrote {OUTPUT_DIR / 'infer_onnx.py'}")

print("\nBundle contents:")
for p in sorted(OUTPUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUTPUT_DIR)}  ({p.stat().st_size / 1e3:.1f} kB)")

print("\nDone. To run inference:")
print(f"  pip install onnxruntime tiktoken")
print(f"  python {OUTPUT_DIR}/infer_onnx.py")
