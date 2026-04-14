"""Utilities for training, inference, data loading, and analysis."""

# Weight initialization
from .weight_initializer import (
    xavier_uniform,
    xavier_normal,
    he_uniform,
    he_normal,
    normal,
    uniform,
    zeros,
    ones,
    orthogonal,
)

# Inference utilities
from .inference import (
    greedy_decode,
    beam_search,
    temperature_sampling,
    top_k_sampling,
    top_p_sampling,
)

# Checkpointing
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_checkpoint_info,
    find_latest_checkpoint,
    compare_checkpoints,
)

# Data loading and preprocessing
from .data import (
    Dataset,
    DataLoader,
    pad_sequences,
    create_batches,
    create_autoregressive_dataset,
    create_mask_for_padding,
    create_causal_mask,
    batch_size_aware_pack,
)

# Metrics tracking and logging
from .metrics import (
    MetricsTracker,
    accuracy,
    perplexity,
    bleu_score,
    f1_score,
    top_k_accuracy,
)

# Visualization helpers
from .visualizers import (
    plot_attention_heads,
    plot_embeddings,
    plot_loss_curve,
    plot_gradient_flow,
    plot_token_distribution,
    compare_attention_patterns,
)

__all__ = [
    # Weight initialization
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "normal",
    "uniform",
    "zeros",
    "ones",
    "orthogonal",
    # Inference
    "greedy_decode",
    "beam_search",
    "temperature_sampling",
    "top_k_sampling",
    "top_p_sampling",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "get_checkpoint_info",
    "find_latest_checkpoint",
    "compare_checkpoints",
    # Data loading
    "Dataset",
    "DataLoader",
    "pad_sequences",
    "create_batches",
    "create_autoregressive_dataset",
    "create_mask_for_padding",
    "create_causal_mask",
    "batch_size_aware_pack",
    # Metrics
    "MetricsTracker",
    "accuracy",
    "perplexity",
    "bleu_score",
    "f1_score",
    "top_k_accuracy",
    # Visualization
    "plot_attention_heads",
    "plot_embeddings",
    "plot_loss_curve",
    "plot_gradient_flow",
    "plot_token_distribution",
    "compare_attention_patterns",
]
