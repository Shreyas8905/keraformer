"""Padding attention masks."""

from __future__ import annotations


def padding_mask(token_ids: list[list[int]], pad_token_id: int = 0) -> list[list[list[list[float]]]]:
    """Return an additive padding mask of shape (B, 1, 1, T).

    Entries that match ``pad_token_id`` are set to negative infinity; all
    other positions are 0.0.
    """

    mask: list[list[list[list[float]]]] = []
    for sequence in token_ids:
        batch_item = [[[float("-inf") if token == pad_token_id else 0.0 for token in sequence]]]
        mask.append(batch_item)
    return mask
