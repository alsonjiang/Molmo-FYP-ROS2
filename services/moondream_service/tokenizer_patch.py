# services/moondream_service/tokenizer_patch.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from transformers import AutoTokenizer

try:
    import tokenizers
except Exception:
    tokenizers = None


@dataclass
class _Encoding:
    ids: List[int]
    tokens: Optional[List[str]] = None


class HFTokenizerAdapter:
    """
    Adapter that mimics the tiny subset of `tokenizers.Tokenizer` that Moondream
    typically uses: encode().ids and decode(ids).
    Backed by a Transformers slow tokenizer (pure Python path).
    """
    def __init__(self, hf_tok):
        self.hf = hf_tok

    def encode(self, text: str) -> _Encoding:
        out = self.hf(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        ids = out["input_ids"]
        # Some HF tokenizers return list[int] for single string, some return list[list[int]]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return _Encoding(ids=ids)

    def decode(self, ids: List[int]) -> str:
        return self.hf.decode(ids, skip_special_tokens=False)

    # Optional helpers in case remote code calls them
    def token_to_id(self, token: str) -> Optional[int]:
        vid = self.hf.convert_tokens_to_ids(token)
        return None if vid == self.hf.unk_token_id else int(vid)

    def id_to_token(self, idx: int) -> str:
        return self.hf.convert_ids_to_tokens(int(idx))

    def get_vocab(self) -> Dict[str, int]:
        return self.hf.get_vocab()


def apply_moondream_tokenizer_patch(
    target_repo: str = "moondream/starmie-v1",
    slow_tokenizer_repo: str = "moondream/starmie-v1",
):
    """
    Monkeypatch `tokenizers.Tokenizer.from_pretrained` so when Moondream tries to
    load its tokenizer via Rust tokenizers, we instead supply a HF slow tokenizer adapter.
    """
    if tokenizers is None:
        raise RuntimeError("tokenizers package not importable")

    orig = tokenizers.Tokenizer.from_pretrained

    def patched(name: str, *args: Any, **kwargs: Any):
        if name == target_repo:
            hf_tok = AutoTokenizer.from_pretrained(slow_tokenizer_repo, use_fast=False)
            return HFTokenizerAdapter(hf_tok)
        return orig(name, *args, **kwargs)

    tokenizers.Tokenizer.from_pretrained = patched
