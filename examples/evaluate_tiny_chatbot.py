"""evaluate the tiny chatbot checkpoint on toy prompts."""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

from keraformer.models import GPT
from keraformer.utils import accuracy, load_checkpoint, perplexity

from examples.common import SimpleTokenizer, build_next_token_dataset


def toy_eval_texts() -> List[str]:
    return [
        "hello how are you",
        "what is your name",
        "tell me a joke",
        "goodbye see you soon",
    ]


def load_vocab(tokenizer: SimpleTokenizer, vocab_path: str) -> None:
    token_to_id = {}
    id_to_token = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            idx_s, tok = line.rstrip("\n").split("\t", 1)
            idx = int(idx_s)
            token_to_id[tok] = idx
            id_to_token[idx] = tok
    tokenizer.token_to_id = token_to_id
    tokenizer.id_to_token = id_to_token


def evaluate(args: argparse.Namespace) -> None:
    ckpt = load_checkpoint(args.checkpoint)
    weights = ckpt["weights"]
    metadata = ckpt.get("metadata", {})

    tokenizer = SimpleTokenizer()
    load_vocab(tokenizer, args.vocab)

    d_model = int(metadata.get("d_model", 64))
    num_heads = int(metadata.get("num_heads", 4))
    num_layers = int(metadata.get("num_layers", 2))
    block_size = int(metadata.get("block_size", 12))

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        tie_weights=False,
        seed=args.seed,
    )
    model.embed = weights["embed"]
    model.lm_head = weights["lm_head"]

    x_eval, y_eval = build_next_token_dataset(toy_eval_texts(), tokenizer, block_size=block_size)
    if x_eval.shape[0] == 0:
        raise RuntimeError("No evaluation samples were produced")

    logits = model.call(x_eval, training=False)
    last_logits = logits[:, -1, :]
    pred = np.argmax(last_logits, axis=-1)

    acc = accuracy(pred, y_eval)
    ppl = perplexity(last_logits[:, None, :], y_eval[:, None])

    print(f"Eval samples: {x_eval.shape[0]}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Perplexity: {ppl:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tiny chatbot checkpoint")
    parser.add_argument("--checkpoint", type=str, default="artifacts/tiny_chatbot_phase11.npz")
    parser.add_argument("--vocab", type=str, default="artifacts/toy_vocab.txt")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
