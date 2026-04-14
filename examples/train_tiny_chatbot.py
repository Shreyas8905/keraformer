"""Phase 11 example: train a tiny next-token chatbot head end-to-end.

Flow:
1) Data ingestion from a toy dialogue corpus
2) Tokenization + dataset construction
3) Train GPT lm_head with AdamW (decoder frozen)
4) Track metrics and save checkpoint
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

from keraformer.models import GPT
from keraformer.optimizers import AdamW
from keraformer.utils import MetricsTracker, save_checkpoint

from examples.common import SimpleTokenizer, build_next_token_dataset, gpt_hidden_and_logits, softmax


def toy_dialogues() -> List[str]:
    return [
        "hello how are you",
        "i am fine thanks",
        "what is your name",
        "my name is keraformer bot",
        "can you help me",
        "yes i can help",
        "tell me a joke",
        "why did the model cross the road",
        "to optimize the loss",
        "goodbye see you soon",
    ]


def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    texts = toy_dialogues()
    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)

    x_train, y_train = build_next_token_dataset(texts, tokenizer, block_size=args.block_size)
    if x_train.shape[0] == 0:
        raise RuntimeError("Empty dataset built from input texts")

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        tie_weights=False,
        seed=args.seed,
    )

    optimizer = AdamW(lr=args.lr, weight_decay=args.weight_decay)
    tracker = MetricsTracker(use_mlflow=args.use_mlflow, experiment_name="keraformer_phase11")

    indices = np.arange(x_train.shape[0])
    for step in range(1, args.steps + 1):
        batch_ids = np.random.choice(indices, size=min(args.batch_size, len(indices)), replace=False)
        x_batch = x_train[batch_ids]
        y_batch = y_train[batch_ids]

        hidden, logits = gpt_hidden_and_logits(model, x_batch)
        last_hidden = hidden[:, -1, :]  # (B, D)
        last_logits = logits[:, -1, :]  # (B, V)

        probs = softmax(last_logits, axis=-1)
        one_hot = np.eye(tokenizer.vocab_size, dtype=np.float32)[y_batch]

        ce = -np.sum(one_hot * np.log(np.maximum(probs, 1e-9)), axis=-1)
        loss = float(np.mean(ce))
        pred = np.argmax(last_logits, axis=-1)
        acc = float(np.mean(pred == y_batch))

        grad_logits = (probs - one_hot) / x_batch.shape[0]  # (B, V)
        grad_lm_head = np.matmul(last_hidden.T, grad_logits).astype(np.float32)  # (D, V)
        model.lm_head = optimizer.step(model.lm_head, grad_lm_head)

        tracker.update(step=step, train_loss=loss, train_accuracy=acc)
        if step % args.log_every == 0 or step == 1:
            print(f"step={step:04d} loss={loss:.4f} acc={acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "tiny_chatbot_phase11.npz")
    save_checkpoint(
        ckpt_path,
        weights={
            "embed": model.embed,
            "lm_head": model.lm_head,
        },
        metadata={
            "vocab_size": tokenizer.vocab_size,
            "block_size": args.block_size,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
        },
        step=args.steps,
    )

    vocab_path = os.path.join(args.out_dir, "toy_vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for idx in range(tokenizer.vocab_size):
            f.write(f"{idx}\t{tokenizer.id_to_token[idx]}\n")

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved vocab: {vocab_path}")
    tracker.end_run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny chatbot example")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--use-mlflow", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
