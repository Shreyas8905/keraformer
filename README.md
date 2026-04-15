# keraformer

Research-grade transformer building blocks for fast experimentation, clear math ownership, and composable model design.

> "I faced the problem so solved it."

Created and maintained by [Shreyas8905](https://github.com/Shreyas8905).

## Executive Summary

Most deep learning teams eventually hit the same friction point: mature frameworks are excellent for training and deployment, but difficult to adapt when you need direct control over internals. `keraformer` is built to address that gap.

This repository provides a modular transformer stack where each component can be inspected, replaced, and extended independently. Instead of forcing researchers to reimplement complete models for every experiment, `keraformer` enables targeted iteration across attention, normalization, feedforward, losses, optimizers, and utility workflows.

## Problem Statement

In practical LLM and custom DL research, common pain points include:

- high overhead when testing new math ideas,
- tightly coupled architecture code that slows ablation studies,
- opaque implementation paths that reduce reproducibility,
- fragmented tooling between model design, training, evaluation, and inference.

## Solution Strategy

`keraformer` applies a block-first strategy:

1. Implement transformer math primitives as independent modules.
2. Compose them into architecture blocks.
3. Build model families on top of shared blocks.
4. Support full experimentation with optimizers, losses, utilities, and end-to-end examples.

This makes it easier to prototype new model ideas without sacrificing readability or scientific control.

## Core Capabilities

### Transformer Building Blocks

- Masks: causal, padding, prefix-LM.
- Embeddings: token embedding, sinusoidal, learned, RoPE, ALiBi, relative, no positional encoding.
- Normalization: LayerNorm, RMSNorm, GroupNorm, DeepNorm.
- Attention: scaled dot-product, MHA, MQA, GQA, MLA, sliding-window, linear attention, cross-attention, flash-style wrapper.
- Feedforward: FFN, Gated FFN variants, MoE FFN, Conv FFN.

### Model Wrappers

- Transformer
- BERT
- GPT
- T5
- Vision Transformer

### Optimization and Losses

- Optimizers and schedules: AdamW, Lion, Adafactor, Noam schedule.
- Losses: label-smoothed cross entropy, focal loss, NT-Xent, masked LM loss.

### Research Utilities

- Weight initialization helpers.
- Inference and decoding helpers: greedy, beam, temperature, top-k, top-p.
- Checkpoint management: save, load, compare, inspect latest.
- Data helpers: sequence dataset/dataloader, batching, masks.
- Metrics tracker with optional MLflow support.
- Visualization helpers for attention, embeddings, gradients, and token statistics.

### End-to-End Example Flow

The example pipeline demonstrates full usage from ingestion to chat interaction:

- [examples/common.py](examples/common.py)
- [examples/train_tiny_chatbot.py](examples/train_tiny_chatbot.py)
- [examples/evaluate_tiny_chatbot.py](examples/evaluate_tiny_chatbot.py)
- [examples/chat_tiny_chatbot.py](examples/chat_tiny_chatbot.py)

## Why This Approach Is Valuable

Compared to many existing solutions, `keraformer` is intentionally optimized for modular experimentation.

- Better component isolation for controlled ablations.
- Lower modification cost for custom math or architecture changes.
- Cleaner mapping from research paper concepts to executable code.
- Easier review and reproducibility due to explicit implementation boundaries.
- Strong fit for researchers who require fine-grained control over equations and tensor flows.

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/Shreyas8905/keraformer.git
```

### Clone Source (recommended for advanced researchers)

```bash
git clone https://github.com/Shreyas8905/keraformer.git
cd keraformer
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install editable package:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[tf]
pip install -e .[jax]
pip install -e .[torch]
pip install -e .[dev]
```

## Quick Start

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run tiny chatbot end-to-end:

```bash
python -m examples.train_tiny_chatbot
python -m examples.evaluate_tiny_chatbot
python -m examples.chat_tiny_chatbot
```

### Contributor Lint Check

If you want to lint a single file or a directory before sending a contribution, use the helper below:

```bash
python tests/lint_file.py path/to/file_or_directory
```

The repository also includes a pull request lint gate in `.github/workflows/pr-lint.yml` that runs Ruff on incoming PRs.

## Caution and Contribution Policy

This repository was built by a single developer and may contain bugs, numerical edge cases, and implementation flaws.

Contributions are welcome for:

- bug fixes,
- numerical stability improvements,
- performance enhancements,
- new transformer blocks,
- new math formulations and experimental modules.

### Recommended Issue Template

1. Title: precise, module-specific summary.
2. Environment: OS, Python version, dependency versions.
3. Affected component: file path and symbol/class/function.
4. Expected behavior.
5. Actual behavior.
6. Minimal reproducible example.
7. Logs or traceback.
8. Proposed fix or hypothesis.

## References and Inspirations

The architecture choices and mathematical blocks are informed by the following papers, repositories, and technical resources.

1. Vaswani et al., Attention Is All You Need (2017) - https://arxiv.org/abs/1706.03762
2. Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers (2018) - https://arxiv.org/abs/1810.04805
3. Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2, 2019) - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
4. Brown et al., Language Models are Few-Shot Learners (GPT-3, 2020) - https://arxiv.org/abs/2005.14165
5. Raffel et al., T5 (2019) - https://arxiv.org/abs/1910.10683
6. Dosovitskiy et al., Vision Transformer (2020) - https://arxiv.org/abs/2010.11929
7. Zhang and Sennrich, RMSNorm (2019) - https://arxiv.org/abs/1910.07467
8. Wang et al., DeepNet / DeepNorm (2022) - https://arxiv.org/abs/2203.00555
9. Su et al., RoFormer / RoPE (2021) - https://arxiv.org/abs/2104.09864
10. Press et al., ALiBi (2021) - https://arxiv.org/abs/2108.12409
11. Shaw et al., Relative Position Representations (2018) - https://arxiv.org/abs/1803.02155
12. Shazeer, Multi-Query Attention (2019) - https://arxiv.org/abs/1911.02150
13. Ainslie et al., Grouped-Query Attention (2023) - https://arxiv.org/abs/2305.13245
14. Dao et al., FlashAttention (2022) - https://arxiv.org/abs/2205.14135
15. Dao et al., FlashAttention-2 (2023) - https://arxiv.org/abs/2307.08691
16. Beltagy et al., Longformer (2020) - https://arxiv.org/abs/2004.05150
17. Choromanski et al., Performers (2020) - https://arxiv.org/abs/2009.14794
18. Fedus et al., Switch Transformers (2021) - https://arxiv.org/abs/2101.03961
19. Lepikhin et al., GShard (2020) - https://arxiv.org/abs/2006.16668
20. Loshchilov and Hutter, AdamW (2017) - https://arxiv.org/abs/1711.05101
21. Chen et al., Lion (2023) - https://arxiv.org/abs/2302.06675
22. Shazeer and Stern, Adafactor (2018) - https://arxiv.org/abs/1804.04235
23. Lin et al., Focal Loss (2017) - https://arxiv.org/abs/1708.02002
24. Chen et al., SimCLR / NT-Xent (2020) - https://arxiv.org/abs/2002.05709
25. PyTorch repository and docs - https://github.com/pytorch/pytorch
26. Keras repository and docs - https://github.com/keras-team/keras
27. Hugging Face Transformers - https://github.com/huggingface/transformers
28. xFormers repository - https://github.com/facebookresearch/xformers
29. Stanford CS25 transformer resources - https://web.stanford.edu/class/cs25/
30. The Illustrated Transformer article - https://jalammar.github.io/illustrated-transformer/
