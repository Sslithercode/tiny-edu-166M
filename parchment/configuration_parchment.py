from transformers import PretrainedConfig


class ParchmentConfig(PretrainedConfig):
    model_type = "parchment"

    def __init__(
        self,
        vocab_size: int = 100277,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_seq_len: int = 1024,
        rms_norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        # aliases expected by transformers internals
        self.num_hidden_layers = n_layers
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
