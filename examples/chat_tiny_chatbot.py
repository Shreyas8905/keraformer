"""interactive chat loop using trained tiny checkpoint."""

from __future__ import annotations

import argparse

import numpy as np

from keraformer.models import GPT
from keraformer.utils import load_checkpoint, top_k_sampling, top_p_sampling

from examples.common import SimpleTokenizer


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


def generate_reply(
    model: GPT,
    tokenizer: SimpleTokenizer,
    prompt: str,
    block_size: int,
    max_new_tokens: int = 12,
    sampling: str = "top_p",
) -> str:
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)

    for _ in range(max_new_tokens):
        if len(token_ids) > block_size:
            context = token_ids[-block_size:]
        else:
            context = token_ids

        row = np.full((1, block_size), tokenizer.pad_id, dtype=np.int64)
        row[0, -len(context) :] = np.asarray(context, dtype=np.int64)

        logits = model.call(row, training=False)
        next_logits = logits[0, -1, :]

        if sampling == "top_k":
            next_id = int(top_k_sampling(next_logits, k=min(8, tokenizer.vocab_size), temperature=1.0)[0])
        else:
            sampled = top_p_sampling(next_logits, p=0.9, temperature=1.0)
            next_id = int(sampled[0] if sampled.ndim > 0 else sampled)

        token_ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break

    return tokenizer.decode(token_ids)


def run_chat(args: argparse.Namespace) -> None:
    ckpt = load_checkpoint(args.checkpoint)
    weights = ckpt["weights"]
    metadata = ckpt.get("metadata", {})

    tokenizer = SimpleTokenizer()
    load_vocab(tokenizer, args.vocab)

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=int(metadata.get("d_model", 64)),
        num_heads=int(metadata.get("num_heads", 4)),
        num_layers=int(metadata.get("num_layers", 2)),
        tie_weights=False,
        seed=args.seed,
    )
    model.embed = weights["embed"]
    model.lm_head = weights["lm_head"]

    block_size = int(metadata.get("block_size", 12))

    print("Tiny keraformer chatbot. Type 'exit' to stop.")
    while True:
        user = input("you> ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("bot> goodbye")
            break
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=user,
            block_size=block_size,
            max_new_tokens=args.max_new_tokens,
            sampling=args.sampling,
        )
        print(f"bot> {reply}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with tiny chatbot checkpoint")
    parser.add_argument("--checkpoint", type=str, default="artifacts/tiny_chatbot_phase11.npz")
    parser.add_argument("--vocab", type=str, default="artifacts/toy_vocab.txt")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--sampling", type=str, choices=["top_k", "top_p"], default="top_p")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run_chat(parse_args())
