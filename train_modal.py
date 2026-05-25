"""
Transformer pretraining on FineWeb-Edu using Modal + wandb.

Setup:
    pip install modal wandb
    modal token new
    wandb login
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
    modal secret create huggingface-secret HF_TOKEN=<your-token>

Run:
    modal run train_modal.py --preprocess-only   # tokenize once first (~2hrs, no GPU)
    modal run train_modal.py                     # preprocess if needed, then train
    modal run train_modal.py --resume ckpt_002000.pt  # force resume from specific ckpt
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "datasets>=2.21.0",
        "tiktoken",
        "wandb",
        "huggingface_hub>=0.24.0",
        "transformers",
        "numpy",
        "pyarrow",
    )
)

app = modal.App("transformer-pretrain", image=image)

volume         = modal.Volume.from_name("pretrain-vol", create_if_missing=True)
MOUNT          = "/vol"
DATA_PATH      = f"{MOUNT}/fineweb_edu_10BT.bin"
CHECKPOINT_DIR = f"{MOUNT}/checkpoints"
META_PATH      = f"{CHECKPOINT_DIR}/meta.json"

# Config 

config = {
    # model
    "vocab_size":       100277,
    "d_model":          768,
    "n_heads":          12,
    "n_layers":         12,
    "d_ff":             3072,
    "max_seq_len":      1024,
    # training
    "batch_size":       16,
    "grad_accum_steps": 8,            # effective batch ~147k tokens
    "lr":               3e-4,
    "min_lr":           3e-5,
    "weight_decay":     0.1,
    "max_steps":        76300,
    "warmup_steps":     2000,
    "grad_clip":        1.0,
    # logging / checkpointing
    "log_every":        100,
    "val_every":        500,
    "val_samples":      2048,         # FIX 4: 256 → 2048 for lower-variance val loss
    "save_every":       2000,
    "keep_last_n":      3,
    "gpu_tflops":       312e12,
    "wandb_project":    "transformer-pretrain-llm-scratch",
    "wandb_run_name":   "fineweb-edu-125M-8xA100",
}


# module-level so multiprocessing can pickle it
def _tokenize_file(args):
    filename, local_path, data_path = args
    import tiktoken
    import numpy as np
    import pyarrow.parquet as pq
    import os

    enc        = tiktoken.get_encoding("cl100k_base")
    eot        = enc.eot_token
    assert eot < 100277, f"eot_token {eot} >= vocab_size 100277"
    assert enc.n_vocab == 100277, f"unexpected n_vocab {enc.n_vocab}, expected 100277"
    shard_path = data_path + f".{filename}"
    tmp        = shard_path + ".tmp"

    if os.path.exists(shard_path):
        print(f"  {filename} shard already exists — skipping", flush=True)
        return filename, shard_path, os.path.getsize(shard_path) // 4

    print(f"  tokenizing {filename}...", flush=True)
    table  = pq.read_table(local_path, columns=["text"])
    texts  = table["text"].to_pylist()
    total  = 0
    BATCH  = 5000
    BUF    = 2_000_000

    with open(tmp, "wb") as f:
        buf = []
        for i in range(0, len(texts), BATCH):
            batch        = texts[i:i + BATCH]
            batch_tokens = enc.encode_ordinary_batch(batch, num_threads=2)
            for tokens in batch_tokens:
                tokens.append(eot)
                buf.extend(tokens)
                total += len(tokens)

            if len(buf) >= BUF:
                np.array(buf, dtype=np.uint32).tofile(f)
                buf = []

            if i % 500000 == 0 and i > 0:
                print(f"  {filename} | docs={i:,} | tokens={total/1e9:.3f}B", flush=True)

        if buf:
            np.array(buf, dtype=np.uint32).tofile(f)

    os.rename(tmp, shard_path)
    print(f"  {filename} DONE: {total/1e9:.3f}B tokens", flush=True)
    return filename, shard_path, total


# ── Preprocess ─────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
    timeout=60 * 60 * 6,
    cpu=16,
)
def preprocess():
    import tiktoken
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import Pool
    import os

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        size_gb = os.path.getsize(DATA_PATH) / 1e9
        print(f"Already exists: {DATA_PATH} ({size_gb:.1f}GB) — skipping.")
        return

    hf_token      = os.environ["HF_TOKEN"]
    PARQUET_DIR   = "/tmp/parquet"
    PARQUET_FILES = [f"{i:03d}_00000" for i in range(13)]
    os.makedirs(PARQUET_DIR, exist_ok=True)

    def download_file(filename):
        local_path = os.path.join(PARQUET_DIR, f"{filename}.parquet")
        if os.path.exists(local_path):
            print(f"  {filename} already downloaded — skipping", flush=True)
            return local_path
        print(f"  downloading {filename}.parquet...", flush=True)
        hf_hub_download(
            repo_id="HuggingFaceFW/fineweb-edu",
            filename=f"sample/10BT/{filename}.parquet",
            repo_type="dataset",
            token=hf_token,
            local_dir=PARQUET_DIR,
        )
        for root, dirs, files in os.walk(PARQUET_DIR):
            for f in files:
                if f == f"{filename}.parquet":
                    found = os.path.join(root, f)
                    if found != local_path:
                        os.rename(found, local_path)
        print(f"  {filename} downloaded", flush=True)
        return local_path

    print(f"Phase 1: downloading {len(PARQUET_FILES)} parquet files in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        local_paths = list(executor.map(download_file, PARQUET_FILES))
    print("All files downloaded.")

    print(f"Phase 2: tokenizing {len(PARQUET_FILES)} files with multiprocessing...")
    args = [(f, p, DATA_PATH) for f, p in zip(PARQUET_FILES, local_paths)]

    with Pool(processes=6) as pool:
        results_list = list(pool.imap_unordered(_tokenize_file, args, chunksize=1))

    results = {r[0]: (r[1], r[2]) for r in results_list}
    print("All files tokenized.")

    print("Phase 3: concatenating...")
    tmp_path     = DATA_PATH + ".tmp"
    total_tokens = 0

    with open(tmp_path, "wb") as out:
        for filename in PARQUET_FILES:
            shard_path, shard_tokens = results[filename]
            with open(shard_path, "rb") as f:
                while chunk := f.read(64 * 1024 * 1024):
                    out.write(chunk)
            total_tokens += shard_tokens
            os.remove(shard_path)
            print(f"  merged {filename}", flush=True)

    os.rename(tmp_path, DATA_PATH)
    volume.commit()
    print(f"Done. {total_tokens/1e9:.2f}B tokens | {total_tokens*4/1e9:.1f}GB on disk.")


# Train function
@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 20,   # 20hrs — ~5B tokens at ~90k tok/s on single A100
    volumes={MOUNT: volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    memory=65536,
    ephemeral_disk=524288,
)
def train(resume_from: str = None):
    _train_worker(resume_from=resume_from)


def _train_worker(resume_from: str = None): # called train worker incase I want to go multi gpu later
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import wandb
    import math
    import time
    import os
    import glob
    import json

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    # Single GPU setup 

    rank       = 0
    world_size = 1
    is_master  = True

    #  Model 

    class Embeddings(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.embeds = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(self.embeds.weight, mean=0, std=d_model ** -0.5)

        def forward(self, token_ids):
            return self.embeds(token_ids)

    class RoPE(nn.Module):
        def __init__(self, d_k, max_seq_len=1024):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))
            self.register_buffer("inv_freq", inv_freq)
            self._build_cache(max_seq_len)

        def _build_cache(self, seq_len):
            t     = torch.arange(seq_len, device=self.inv_freq.device).float()
            freqs = torch.outer(t, self.inv_freq)
            emb   = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos_cache", emb.cos()[None, None])
            self.register_buffer("sin_cache", emb.sin()[None, None])

        def rotate_half(self, x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        def forward(self, q, k):
            seq = q.shape[2]
            assert seq <= self.cos_cache.shape[2], \
                f"sequence length {seq} exceeds RoPE cache size {self.cos_cache.shape[2]}"
            cos = self.cos_cache[:, :, :seq]
            sin = self.sin_cache[:, :, :seq]
            q   = (q * cos) + (self.rotate_half(q) * sin)
            k   = (k * cos) + (self.rotate_half(k) * sin)
            return q, k

    class RMSNorm(nn.Module):
        def __init__(self, d_model, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d_model))
            self.eps   = eps

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

    class MultiHeadAttention(nn.Module):
        def __init__(self, n_heads, d_model):
            super().__init__()
            self.n_heads = n_heads
            self.d_k     = d_model // n_heads
            self.d_model = d_model
            self.W_Q = nn.Linear(d_model, d_model, bias=False)
            self.W_K = nn.Linear(d_model, d_model, bias=False)
            self.W_V = nn.Linear(d_model, d_model, bias=False)
            self.W_O = nn.Linear(d_model, d_model, bias=False)
            self.rope = RoPE(self.d_k, max_seq_len=config["max_seq_len"])
            self.W_O.RESIDUAL_SCALE_INIT = True

        def forward(self, x):
            B, T, _ = x.shape
            Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
            K = self.W_K(x).view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
            V = self.W_V(x).view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
            Q, K = self.rope(Q, K)
            out  = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            out  = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
            return self.W_O(out)

    class SwiGLU(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            hidden  = int(2/3 * 4 * d_model)
            hidden  = (hidden + 63) // 64 * 64
            self.w1 = nn.Linear(d_model, hidden, bias=False)
            self.w2 = nn.Linear(hidden,  d_model, bias=False)
            self.w3 = nn.Linear(d_model, hidden, bias=False)
            self.w2.RESIDUAL_SCALE_INIT = True

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff):
            super().__init__()
            self.attn  = MultiHeadAttention(n_heads, d_model)
            self.ff    = SwiGLU(d_model)
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x

    class Transformer(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.embeddings = Embeddings(cfg["vocab_size"], cfg["d_model"])
            self.blocks     = nn.ModuleList([
                TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_ff"])
                for _ in range(cfg["n_layers"])
            ])
            self.norm = RMSNorm(cfg["d_model"])
            self.head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
            self.head.weight = self.embeddings.embeds.weight
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                std = 0.02
                if hasattr(module, "RESIDUAL_SCALE_INIT"):
                    std /= math.sqrt(2 * config["n_layers"])
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, token_ids):
            x = self.embeddings(token_ids)
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm(x))

    # Dataset 
    class MmapDataset(Dataset):
        def __init__(self, path, block_size, start_sample=0, max_samples=None, stride=1):
            data         = np.memmap(path, dtype=np.uint32, mode="r")
            self.data    = data
            self.block_size = block_size
            total        = (len(data) - 1) // block_size
            # stride > 1 used for per-rank DDP splits; max_samples caps val set
            self.indices = range(start_sample, total, stride)
            if max_samples is not None:
                self.indices = self.indices[:max_samples]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i     = self.indices[idx] * self.block_size
            chunk = self.data[i : i + self.block_size + 1]
            x     = torch.from_numpy(chunk[:-1].astype(np.int64))
            y     = torch.from_numpy(chunk[1:].astype(np.int64))
            return x, y

    #  LR schedule 
    def get_lr(step):
        if step < config["warmup_steps"]:
            return config["lr"] * (step + 1) / config["warmup_steps"]
        progress = (step - config["warmup_steps"]) / (config["max_steps"] - config["warmup_steps"])
        return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))

    #  Meta helpers
    def load_meta():
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                return json.load(f)
        return {}

    def save_meta(meta):
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)

    # Checkpoint helpers
    def save_checkpoint(model, optimizer, step, tokens_seen, loss_val, wandb_run_id):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f"ckpt_{step:06d}.pt")

        raw_model = model.module if hasattr(model, "module") else model
        raw_sd    = raw_model._orig_mod.state_dict() if hasattr(raw_model, "_orig_mod") else raw_model.state_dict()

        torch.save({
            "step":              step,
            "tokens_seen":       tokens_seen,
            # first sample index for the next training run — skips val_samples
            # held-out at the front of the file, then advances by everything
            # consumed so far. used as start_sample on resume.
            "train_sample_next": step * config["batch_size"] * config["grad_accum_steps"] + config["val_samples"],
            "loss":              loss_val,
            "model":             raw_sd,
            "optimizer":         optimizer.state_dict(),
            "config":            config,
        }, path)

        save_meta({
            "wandb_run_id":   wandb_run_id,
            "latest_ckpt":    f"ckpt_{step:06d}.pt",
            "step":           step,
            "tokens_seen_B":  tokens_seen / 1e9,
            "loss":           loss_val,
        })

        ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_*.pt")))
        for old in ckpts[: -config["keep_last_n"]]:
            os.remove(old)
            print(f"  → deleted {os.path.basename(old)}")

        volume.commit()
        print(f"  → saved {path} | loss={loss_val:.4f} | tokens={tokens_seen/1e9:.3f}B")
        return path

    #  wandb init

    meta           = load_meta()
    wandb_run_id   = meta.get("wandb_run_id", None)

    if is_master:
        run = wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            id=wandb_run_id,
            config=config,
            resume="allow",
        )
        if wandb_run_id is None:
            wandb_run_id = run.id
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            save_meta({"wandb_run_id": wandb_run_id})
            volume.commit()

    # Model + optimizer

    model = Transformer(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_master:
        print(f"Parameters: {n_params/1e6:.1f}M")

    decay    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = AdamW(
        [{"params": decay,    "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config["lr"],
        betas=(0.9, 0.95),
        fused=True,
    )

    # Resume 
    start_step   = 0
    VAL_SAMPLES  = config["val_samples"]
    start_sample = VAL_SAMPLES

    if resume_from:
        ckpt_path = os.path.join(CHECKPOINT_DIR, resume_from)
        if is_master:
            print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step   = ckpt["step"]
        start_sample = ckpt.get(
            "train_sample_next",
            # fallback for old checkpoints that used "samples_seen"
            ckpt.get(
                "samples_seen",
                start_step * config["batch_size"] * config["grad_accum_steps"] + VAL_SAMPLES,
            ),
        )
        if is_master:
            print(f"  step={start_step} | start_sample={start_sample:,} | loss={ckpt.get('loss', '?')}")

    model = torch.compile(model)

    wandb.run.summary["n_params"] = n_params

    # Copy dataset to local NVMe 
    
    import shutil
    LOCAL_DATA = "/tmp/fineweb_edu_10BT.bin"
    if not os.path.exists(LOCAL_DATA):
        if is_master:
            print(f"Copying dataset to local NVMe ({os.path.getsize(DATA_PATH)/1e9:.1f}GB)...")
        t_copy = time.perf_counter()
        shutil.copy2(DATA_PATH, LOCAL_DATA)
        if is_master:
            print(f"  copied in {time.perf_counter() - t_copy:.1f}s")
    else:
        if is_master:
            print("Local data already exists — skipping copy.")

    #  Datasets 

    _tmp_data    = np.memmap(LOCAL_DATA, dtype=np.uint32, mode="r")
    total_samples = (len(_tmp_data) - 1) // config["max_seq_len"]
    del _tmp_data
    print(f"Dataset: {total_samples:,} total samples")

    def make_loader(path, start, max_s=None, batch_size=config["batch_size"]):
        ds = MmapDataset(path, config["max_seq_len"],
                         start_sample=start, max_samples=max_s)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          pin_memory=True, num_workers=0)

    val_loader   = make_loader(LOCAL_DATA, start=0, max_s=VAL_SAMPLES)
    train_loader = make_loader(LOCAL_DATA, start=start_sample)

    def run_validation():
        model.eval()
        total_loss   = 0.0
        total_tokens = 0
        with torch.inference_mode():
            for vx, vy in val_loader:
                vx = vx.to(device, non_blocking=True)
                vy = vy.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits       = model(vx)
                    n_tokens     = vx.shape[0] * vx.shape[1]
                    total_loss  += F.cross_entropy(
                        logits.view(-1, config["vocab_size"]), vy.view(-1)
                    ).item() * n_tokens
                    total_tokens += n_tokens
        model.train()
        return total_loss / total_tokens

    #  Training loop 

    model.train()
    global_step = start_step
    accum_step  = 0
    accum_loss  = 0.0
    last_loss   = 0.0
    tokens_seen = (
        start_step
        * config["batch_size"]
        * config["grad_accum_steps"]
        * config["max_seq_len"]
    )
    t0 = time.perf_counter()

    _data_time  = 0.0
    _step_time  = 0.0
    _t_batch    = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    for x, y in train_loader:
        if global_step >= config["max_steps"]:
            break

        _data_time += time.perf_counter() - _t_batch

        _t_fwd = time.perf_counter()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss   = F.cross_entropy(
                logits.view(-1, config["vocab_size"]),
                y.view(-1),
            ) / config["grad_accum_steps"]

        loss.backward()
        _step_time += time.perf_counter() - _t_fwd

        accum_loss += loss.item()
        accum_step += 1

        if accum_step < config["grad_accum_steps"]:
            _t_batch = time.perf_counter()
            continue

        #  optimizer step 
        lr = get_lr(global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        accum_step   = 0
        last_loss    = accum_loss
        accum_loss   = 0.0
        _t_batch     = time.perf_counter()

        tokens_seen = (
            global_step
            * config["batch_size"]
            * config["grad_accum_steps"]
            * config["max_seq_len"]
        )

        # logging 
        if global_step % config["log_every"] == 0 and is_master:
            t1  = time.perf_counter()
            dt  = t1 - t0
            t0  = t1

            ppl      = math.exp(min(last_loss, 20))
            tok_s    = (
                config["batch_size"] *
                config["max_seq_len"] *
                config["grad_accum_steps"] *
                config["log_every"] / dt
            )
            mfu      = 6 * n_params * tok_s / config.get("gpu_tflops", 312e12)
            total_t  = _data_time + _step_time
            data_pct = _data_time / total_t if total_t > 0 else 0.0
            _data_time = 0.0
            _step_time = 0.0

            grad_norm_f = float(grad_norm)

            print(
                f"step {global_step:>6} | "
                f"loss {last_loss:.4f} | "
                f"ppl {ppl:.1f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm_f:.3f} | "
                f"tok/s {tok_s:,.0f} | "
                f"MFU {mfu:.1%} | "
                f"data% {data_pct:.1%} | "
                f"tokens {tokens_seen/1e9:.3f}B"
            )

            wandb.log({
                "train/loss":          last_loss,
                "train/perplexity":    ppl,
                "train/lr":            lr,
                "train/grad_norm":     grad_norm_f,
                "perf/tokens_per_sec": tok_s,
                "perf/mfu":            mfu,
                "perf/data_pct":       data_pct,
                "perf/tokens_seen_B":  tokens_seen / 1e9,
            }, step=global_step)

        #  validation 
        if global_step % config["val_every"] == 0:
            val_loss = run_validation()
            val_ppl  = math.exp(min(val_loss, 20))
            print(f"  val loss {val_loss:.4f} | val ppl {val_ppl:.1f}")
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=global_step)

        # checkpoint 

        if global_step % config["save_every"] == 0 and is_master:
            ckpt_path = save_checkpoint(
                model, optimizer,
                global_step, tokens_seen,
                last_loss,
                wandb_run_id,
            )

            if global_step % (config["save_every"] * 5) == 0:
                artifact = wandb.Artifact(
                    name=f"model-step-{global_step}",
                    type="model",
                    metadata={
                        "step":          global_step,
                        "tokens_seen_B": tokens_seen / 1e9,
                        "loss":          last_loss,
                        "perplexity":    math.exp(min(last_loss, 20)),
                    },
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)

    #  Teardown 
    wandb.finish()
    print("Training complete.")


# Entrypoint
@app.local_entrypoint()
def main(resume: str = None, preprocess_only: bool = False):
    """
    modal run train_modal.py --preprocess-only             # tokenize once, no GPU
    modal run train_modal.py                               # auto-resumes if meta.json exists
    modal run train_modal.py --resume ckpt_002000.pt       # force specific checkpoint
    """
    # Use .remote() so errors propagate cleanly and the call blocks until done.
    # .spawn().get() can swallow exceptions on the spawned side.
    preprocess.remote()

    if preprocess_only:
        print("Preprocess complete.")
        return

    import json, os
    meta_local = f"{CHECKPOINT_DIR}/meta.json"
    if resume is None and os.path.exists(meta_local):
        try:
            meta   = json.load(open(meta_local))
            resume = meta.get("latest_ckpt")
            if resume:
                print(f"Auto-resuming from {resume} (step={meta.get('step')}, tokens={meta.get('tokens_seen_B', 0):.2f}B)")
        except Exception:
            pass

    train.spawn(resume_from=resume)
    print("Training job spawned — safe to close terminal. Monitor via wandb.")