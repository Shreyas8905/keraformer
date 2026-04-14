"""Metrics tracking and logging utilities."""

import numpy as np
from typing import Dict, Optional, List, Any
from collections import defaultdict
import warnings
import importlib


class MetricsTracker:
    """Track and log training metrics with optional MLflow integration."""
    
    def __init__(self, use_mlflow: bool = False, experiment_name: Optional[str] = None):
        """Initialize metrics tracker.
        
        Args:
            use_mlflow: Whether to log to MLflow (default False)
            experiment_name: MLflow experiment name (default None)
        """
        self.use_mlflow = use_mlflow
        self.metrics = defaultdict(list)
        self.step = 0
        
        if use_mlflow:
            try:
                mlflow = importlib.import_module("mlflow")
                self.mlflow = mlflow
                if experiment_name:
                    mlflow.set_experiment(experiment_name)
                mlflow.start_run()
            except ImportError:
                warnings.warn("MLflow not installed, disabling MLflow logging")
                self.use_mlflow = False
    
    def update(self, step: int, **metrics: float) -> None:
        """Update metrics for a step.
        
        Args:
            step: Training step number
            **metrics: Named metrics to log (e.g., loss=0.5, accuracy=0.95)
        """
        self.step = step
        
        for name, value in metrics.items():
            self.metrics[name].append((step, value))
            
            if self.use_mlflow:
                try:
                    self.mlflow.log_metric(name, value, step=step)
                except Exception as e:
                    warnings.warn(f"MLflow logging failed for {name}: {e}")
    
    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter (hyperparameter).
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        if self.use_mlflow:
            try:
                self.mlflow.log_param(name, value)
            except Exception as e:
                warnings.warn(f"MLflow param logging failed: {e}")
    
    def log_dict(self, name: str, data: Dict) -> None:
        """Log a dictionary of metrics.
        
        Args:
            name: Base name for metrics
            data: Dictionary of {metric_name: value}
        """
        for key, value in data.items():
            self.update(self.step, **{f"{name}/{key}": value})
    
    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric.
        
        Args:
            name: Metric name
        
        Returns:
            List of metric values
        """
        return [v for _, v in self.metrics.get(name, [])]
    
    def get_metric_at_step(self, name: str, step: int) -> Optional[float]:
        """Get metric value at specific step.
        
        Args:
            name: Metric name
            step: Step number
        
        Returns:
            Metric value or None if not found
        """
        for s, v in self.metrics.get(name, []):
            if s == step:
                return v
        return None
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value of a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Latest value or None if metric not logged
        """
        values = self.metrics.get(name, [])
        return values[-1][1] if values else None
    
    def get_average(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get average of metric values.
        
        Args:
            name: Metric name
            last_n: Average over last n values (default None for all)
        
        Returns:
            Average value or None if metric not logged
        """
        values = [v for _, v in self.metrics.get(name, [])]
        if not values:
            return None
        
        if last_n is not None:
            values = values[-last_n:]
        
        return np.mean(values) if values else None
    
    def end_run(self) -> None:
        """End MLflow run if active."""
        if self.use_mlflow:
            try:
                self.mlflow.end_run()
            except Exception as e:
                warnings.warn(f"MLflow end_run failed: {e}")
    
    def __repr__(self) -> str:
        metrics_str = ", ".join([f"{k}: {self.get_latest(k):.4f}" for k in self.metrics.keys()])
        return f"MetricsTracker(step={self.step}, {metrics_str})"


def accuracy(predictions: np.ndarray, targets: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute token-level accuracy.
    
    Args:
        predictions: Predicted token indices, shape (batch, seq_len)
        targets: Target token indices, shape (batch, seq_len)
        mask: Optional mask, shape (batch, seq_len) with 1 for valid, 0 for padded
    
    Returns:
        Accuracy as float in [0, 1]
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")
    
    correct = (predictions == targets).astype(np.float32)
    
    if mask is not None:
        correct = correct * mask
        total = np.sum(mask)
    else:
        total = predictions.size
    
    return float(np.sum(correct) / total) if total > 0 else 0.0


def perplexity(logits: np.ndarray, targets: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute perplexity from logits.
    
    Args:
        logits: Model logits, shape (batch, seq_len, vocab_size)
        targets: Target token indices, shape (batch, seq_len)
        mask: Optional mask for padding/masking specific positions
    
    Returns:
        Perplexity as float
    """
    # Compute cross-entropy loss
    batch_size, seq_len, vocab_size = logits.shape
    
    # Compute probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Get target probabilities
    targets_flat = targets.reshape(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    
    target_probs = np.maximum(probs.reshape(-1, vocab_size)[np.arange(len(targets_flat)), targets_flat], 1e-10)
    cross_entropy = -np.log(target_probs)
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.reshape(-1)
        cross_entropy = cross_entropy * mask_flat
        total = np.sum(mask_flat)
    else:
        total = len(cross_entropy)
    
    avg_ce = np.sum(cross_entropy) / total if total > 0 else 0.0
    perp = np.exp(avg_ce)
    
    return float(perp)


def bleu_score(predictions: List[List[int]], references: List[List[int]], max_n: int = 4) -> float:
    """Simplified BLEU score computation.
    
    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences
        max_n: Maximum n-gram size (default 4)
    
    Returns:
        BLEU score in [0, 1]
    """
    def get_ngrams(seq: List[int], n: int) -> Dict[tuple, int]:
        ngrams = defaultdict(int)
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    clipped_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    for pred, ref in zip(predictions, references):
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            for ngram, count in pred_ngrams.items():
                clipped_counts[n] += min(count, ref_ngrams.get(ngram, 0))
                total_counts[n] += count
    
    # Compute precision for each n-gram
    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n] > 0:
            p = clipped_counts[n] / total_counts[n]
            precisions.append(max(p, 1e-10))  # Small epsilon to avoid log(0)
        else:
            precisions.append(0.0)
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precisions = np.log(precisions)
    geo_mean = np.exp(np.mean(log_precisions))
    
    # Brevity penalty
    pred_length = sum(len(p) for p in predictions)
    ref_length = sum(len(r) for r in references)
    
    if pred_length == 0 or ref_length == 0:
        return 0.0
    
    brevity_penalty = min(1.0, np.exp(1 - ref_length / pred_length))
    
    bleu = geo_mean * brevity_penalty
    return float(bleu)


def f1_score(predictions: np.ndarray, targets: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute F1 score (macro average).
    
    Args:
        predictions: Predicted labels, shape (batch,) or (batch, seq_len)
        targets: Target labels, shape (batch,) or (batch, seq_len)
        mask: Optional mask for padding
    
    Returns:
        F1 score as float
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")
    
    # Flatten if needed
    if len(predictions.shape) > 1:
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        if mask is not None:
            mask = mask.reshape(-1)
    
    # Get unique classes (excluding background/neutral class if present)
    classes = np.unique(targets)
    
    f1_scores = []
    for cls in classes:
        tp = np.sum((predictions == cls) & (targets == cls))
        fp = np.sum((predictions == cls) & (targets != cls))
        fn = np.sum((predictions != cls) & (targets == cls))
        
        if tp + fp + fn == 0:
            continue
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
        else:
            f1_scores.append(0.0)
    
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def top_k_accuracy(predictions: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy.
    
    Args:
        predictions: Model predictions, shape (batch, num_classes)
        targets: Target labels, shape (batch,)
        k: Consider top-k predictions (default 5)
    
    Returns:
        Top-k accuracy as float in [0, 1]
    """
    if len(predictions.shape) != 2:
        raise ValueError("Predictions must be 2D (batch, num_classes)")
    
    # Get top-k predictions
    top_k_preds = np.argsort(predictions, axis=-1)[:, -k:]
    
    # Count matches
    matches = 0
    for i, target in enumerate(targets):
        if target in top_k_preds[i]:
            matches += 1
    
    return float(matches / len(targets))
