import os
import json
import tiktoken
from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "tiktoken_encoding.json"}


class TiktokenTokenizer(PreTrainedTokenizer):
    """
    HuggingFace-compatible tokenizer wrapping tiktoken's cl100k_base.
    Produces byte-identical token IDs to tiktoken.get_encoding("cl100k_base").
    Tokens are represented internally as their raw bytes decoded via latin-1
    (a lossless bijection for arbitrary byte sequences).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file=None, encoding_name="cl100k_base", **kwargs):
        self.encoding_name = encoding_name
        self._build_enc()

        eot_str = self._id_to_tok[self._enc.eot_token]  # "<|endoftext|>"

        # When loading from saved config, special tokens are already in kwargs —
        # pop them so we don't pass duplicates to super().__init__().
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("pad_token", None)
        kwargs.pop("unk_token", None)

        super().__init__(
            encoding_name=encoding_name,
            bos_token=eot_str,
            eos_token=eot_str,
            pad_token=eot_str,
            unk_token=eot_str,
            **kwargs,
        )

    def _build_enc(self):
        self._enc = tiktoken.get_encoding(self.encoding_name)
        self._id_to_tok = {}
        self._tok_to_id = {}
        for i in range(self._enc.n_vocab):
            try:
                s = self._enc.decode_single_token_bytes(i).decode("latin-1")
            except Exception:
                s = f"<|special_{i}|>"
            self._id_to_tok[i] = s
            self._tok_to_id[s] = i

    # ── Required interface ─────────────────────────────────────────────────────

    @property
    def vocab_size(self):
        return self._enc.n_vocab  # 100277

    def get_vocab(self):
        return dict(self._tok_to_id)

    def _tokenize(self, text):
        ids = self._enc.encode(text, allowed_special="all")
        return [self._id_to_tok[i] for i in ids]

    def _convert_token_to_id(self, token):
        return self._tok_to_id.get(token, self._enc.eot_token)

    def _convert_id_to_token(self, index):
        return self._id_to_tok.get(index, "<|unk|>")

    def convert_tokens_to_string(self, tokens):
        raw = b"".join(t.encode("latin-1") for t in tokens)
        return raw.decode("utf-8", errors="replace")

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """No BOS/EOS added — matches bare tiktoken encode() behaviour."""
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        fname = (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        vocab_file = os.path.join(save_directory, fname)
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump({"encoding_name": self.encoding_name}, f)
        return (vocab_file,)

    # ── Pickle support (tiktoken objects aren't picklable) ─────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_enc", None)
        state.pop("_id_to_tok", None)
        state.pop("_tok_to_id", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_enc()
