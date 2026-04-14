"""Causal attention masks."""

from __future__ import annotations


def causal_mask(seq_len: int) -> list[list[float]]:
    """Return an additive causal mask of shape (seq_len, seq_len).

    The lower triangle, including the diagonal, is visible and encoded as 0.0.
    Future positions are masked with negative infinity.
    """

    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")

    mask: list[list[float]] = []
    for query_index in range(seq_len):
        row: list[float] = []
        for key_index in range(seq_len):
            row.append(0.0 if key_index <= query_index else float("-inf"))
        mask.append(row)
    return mask
