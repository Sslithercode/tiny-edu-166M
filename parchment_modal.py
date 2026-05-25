"""
Parchment — Modal inference endpoint
─────────────────────────────────────
Setup (run once):
    modal volume create parchment-model
    modal volume put parchment-model onnx_bundle/parchment_instruct_fp16.onnx /model/

Deploy:
    modal deploy parchment_modal.py

Test locally:
    modal run parchment_modal.py

Endpoint URL printed after deploy:
    https://your-workspace--parchment-parchment-infer.modal.run
"""

import modal

# ---------------------------------------------------------------------------
# Image — onnxruntime + tiktoken only, no torch, no transformers (~50MB)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime==1.20.1",
        "tiktoken==0.7.0",
        "fastapi[standard]",  # required by modal.fastapi_endpoint
        "numpy",
    )
)

# ---------------------------------------------------------------------------
# Volume — model file lives here, persists across deployments
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("parchment-model")

app = modal.App("parchment", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/model/model/parchment_instruct_fp16.onnx"
EOT_ID     = 100257  # <|endoftext|>
MAX_SEQ    = 1024
VOCAB_SIZE = 100277

# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------
@app.cls(
    cpu=4.0,
    memory=2048,               # 479MB model + ORT + Python overhead
    volumes={"/model": volume},
    scaledown_window=300,      # keep warm 5 min after last request (Modal 1.0)
)
@modal.concurrent(max_inputs=4)  # Modal 1.0: replaces allow_concurrent_inputs
class Parchment:

    @modal.enter()
    def load(self):
        """Runs once when container starts. Session stays resident across requests."""
        import os
        import onnxruntime as ort
        from tiktoken import get_encoding

        print(f"[parchment] loading model ({os.path.getsize(MODEL_PATH) / 1e6:.0f} MB)")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            # ORT_ENABLE_ALL triggers SimplifiedLayerNormFusion which
            # crashes on this fp16 model's Cast nodes
        )
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.enc = get_encoding("cl100k_base")
        print("[parchment] ready")

    # -----------------------------------------------------------------------
    # Chat template
    # -----------------------------------------------------------------------
    def _apply_template(self, system: str, user: str) -> list[int]:
        enc = self.enc

        def role_block(role: str, content: str) -> list[int]:
            return [
                *enc.encode(role + "\n" + content),
                EOT_ID,
                *enc.encode("\n"),
            ]

        return [
            *role_block("system", system),
            *role_block("user", user),
            *enc.encode("assistant\n"),
        ]

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------
    @staticmethod
    def _sample(logits_slice: list, temperature: float, top_k: int) -> int:
        import math
        import random

        temp = max(temperature, 1e-8)
        logits = [v / temp for v in logits_slice]

        if 0 < top_k < VOCAB_SIZE:
            indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
            cutoff = indexed[top_k - 1][1]
            logits = [v if v >= cutoff else -1e9 for v in logits]

        max_l = max(logits)
        exps = [math.exp(v - max_l) for v in logits]
        total = sum(exps)
        probs = [e / total for e in exps]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return VOCAB_SIZE - 1

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    def _generate(self, prompt_ids: list[int], max_new_tokens: int, temperature: float, top_k: int):
        import numpy as np

        tokens = list(prompt_ids)
        prompt_len = len(tokens)
        prev_text_len = 0

        for _ in range(max_new_tokens):
            window = tokens[-MAX_SEQ:]
            input_data = np.array([window], dtype=np.int64)
            result = self.session.run(["logits"], {"input_ids": input_data})

            logits = result[0][0, -1, :].tolist()
            next_id = self._sample(logits, temperature, top_k)
            tokens.append(next_id)

            if next_id == EOT_ID:
                break

            # Decode all new tokens as batch — handles multi-byte UTF-8
            new_bytes = self.enc.decode(tokens[prompt_len:])
            full_text = new_bytes if isinstance(new_bytes, str) else new_bytes.decode("utf-8", errors="replace")
            delta = full_text[prev_text_len:]
            prev_text_len = len(full_text)

            if delta:
                yield delta

    # -----------------------------------------------------------------------
    # Web endpoint — Modal 1.0: modal.fastapi_endpoint (was modal.web_endpoint)
    # StreamingResponse imported from fastapi.responses, not modal.generators
    # -----------------------------------------------------------------------
    @modal.fastapi_endpoint(method="POST", docs=True)
    def infer(self, body: dict):
        from fastapi.responses import StreamingResponse

        message        = body.get("message", "")
        system         = body.get("system", "You are a helpful assistant.")
        max_new_tokens = int(body.get("max_new_tokens", 200))
        temperature    = float(body.get("temperature", 0.8))
        top_k          = int(body.get("top_k", 40))

        if not message:
            return StreamingResponse(
                iter([b"error: message is required"]),
                media_type="text/plain",
                status_code=400,
            )

        prompt_ids = self._apply_template(system, message)

        def stream():
            for chunk in self._generate(prompt_ids, max_new_tokens, temperature, top_k):
                yield chunk.encode("utf-8")

        return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# Local test — modal run parchment_modal.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    p = Parchment()
    prompt = "What is the capital of France?"
    print(f"\n> {prompt}\n")
    for chunk in p.infer.remote({"message": prompt}):
        print(chunk, end="", flush=True)
    print()