"""Prefix language modeling masks."""

from __future__ import annotations


def prefix_lm_mask(seq_len: int, prefix_len: int) -> list[list[float]]:
    """Return an additive prefix-LM mask of shape (seq_len, seq_len).

    Tokens in the prefix can attend bidirectionally within the prefix.
    Tokens in the suffix can attend to the full prefix and to previous suffix
    positions, but not to future suffix positions.
    """

    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if prefix_len < 0:
        raise ValueError("prefix_len must be non-negative")
    if prefix_len > seq_len:
        raise ValueError("prefix_len must be less than or equal to seq_len")

    mask: list[list[float]] = []
    for query_index in range(seq_len):
        row: list[float] = []
        for key_index in range(seq_len):
            visible = key_index < prefix_len or key_index <= query_index
            row.append(0.0 if visible else float("-inf"))
        mask.append(row)
    return mask
