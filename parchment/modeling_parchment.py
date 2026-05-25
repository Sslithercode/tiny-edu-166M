import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_parchment import ParchmentConfig


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embeds = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embeds.weight, mean=0, std=d_model ** -0.5)

    def forward(self, token_ids):
        return self.embeds(token_ids)


class RoPE(nn.Module):
    def __init__(self, d_k, max_seq_len, base=10000.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        # persistent=True so inv_freq is saved/loaded via from_pretrained
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None

    def _build_cache(self, device: torch.device, dtype: torch.dtype):
        t = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device, torch.float32))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos()[None, None].to(dtype)
        self._sin_cache = emb.sin()[None, None].to(dtype)

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        if self._cos_cache is None or self._cos_cache.device != q.device or self._cos_cache.dtype != q.dtype:
            self._build_cache(q.device, q.dtype)
        seq = q.shape[2]
        cos = self._cos_cache[:, :, :seq]
        sin = self._sin_cache[:, :, :seq]
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, max_seq_len, rope_base):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_k, max_seq_len, base=rope_base)
        self.W_O.RESIDUAL_SCALE_INIT = True

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        Q, K = self.rope(Q, K)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
        return self.W_O(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = int(2 / 3 * 4 * d_model)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2.RESIDUAL_SCALE_INIT = True

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, rope_base):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads, d_model, max_seq_len, rope_base)
        self.ff = SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ParchmentModel(PreTrainedModel):
    config_class = ParchmentConfig
    base_model_prefix = "model"

    def __init__(self, config: ParchmentConfig):
        super().__init__(config)
        self.embeddings = Embeddings(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.max_seq_len, config.rope_base)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL_SCALE_INIT"):
                std /= math.sqrt(2 * self.config.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.config.d_model ** -0.5)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ParchmentForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ParchmentConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.embeds.weight"}
    _keys_to_ignore_on_load_missing = [r"lm_head\.weight", r".*\.rope\."]
    _supports_cache_class = False
    _supports_static_cache = False

    def __init__(self, config: ParchmentConfig):
        super().__init__(config)
        self.model = ParchmentModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings.embeds

    def set_input_embeddings(self, value):
        self.model.embeddings.embeds = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _init_weights(self, module):
        self.model._init_weights(module)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden = self.model(input_ids)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids[:, -self.config.max_seq_len:],
            "attention_mask": attention_mask,
        }
