"""Shared helpers for end-to-end examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class SimpleTokenizer:
    """A tiny whitespace tokenizer for toy end-to-end demos."""

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    def __post_init__(self) -> None:
        self.token_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def fit(self, texts: Iterable[str]) -> None:
        for text in texts:
            for token in text.strip().split():
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for token in text.strip().split():
            ids.append(self.token_to_id.get(token, self.token_to_id[self.unk_token]))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        words: List[str] = []
        specials = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if skip_special and token in specials:
                continue
            words.append(token)
        return " ".join(words)


def build_next_token_dataset(
    texts: List[str], tokenizer: SimpleTokenizer, block_size: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """Create fixed-size next-token training pairs from raw texts."""
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) < 2:
            continue
        for i in range(1, len(ids)):
            prefix = ids[:i]
            target = ids[i]
            if len(prefix) > block_size:
                prefix = prefix[-block_size:]

            row = np.full((block_size,), tokenizer.pad_id, dtype=np.int64)
            row[-len(prefix) :] = np.asarray(prefix, dtype=np.int64)
            x_rows.append(row)
            y_rows.append(target)

    if not x_rows:
        return np.zeros((0, block_size), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.int64)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(shifted)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-9)


def gpt_hidden_and_logits(model, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward pass that exposes hidden states before lm_head."""
    ids = np.asarray(token_ids, dtype=np.int64)
    hidden = model.embed[ids]
    for block in model.decoder:
        hidden = block.call(hidden, encoder_output=None, training=True)
    logits = np.matmul(hidden, model.lm_head)
    return hidden.astype(np.float32), logits.astype(np.float32)
