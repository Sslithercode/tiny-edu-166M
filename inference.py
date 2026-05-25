# infer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math

config = {
    "vocab_size":  100277,
    "d_model":     768,
    "n_heads":     12,
    "n_layers":    12,
    "max_seq_len": 1024,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, d_model, n_heads):
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
            TransformerBlock(cfg["d_model"], cfg["n_heads"])
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
    
    
    
model = Transformer(config).to(device)
ckpt  = torch.load("ckpt_030000_final.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()

enc = tiktoken.get_encoding("cl100k_base")

def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    tokens = enc.encode(prompt)
    x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x[:, -config["max_seq_len"]:])[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            x = torch.cat([x, torch.multinomial(logits.softmax(-1), 1)], dim=1)
    return enc.decode(x[0].tolist())

prompt_text = "The history of the united states is"
tokens = enc.encode(prompt_text)
x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, config["vocab_size"]),
        x[:, 1:].reshape(-1),
    )
    print(f"loss: {loss.item():.4f}")  # should be ~3-4