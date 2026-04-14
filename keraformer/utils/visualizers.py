"""Visualization utilities for debugging and analysis."""

import numpy as np
from typing import Optional, List, Dict, Any


def plot_attention_heads(
    attention_weights: np.ndarray,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Weights",
) -> Dict[str, Any]:
    """Prepare attention weight visualization data.
    
    Args:
        attention_weights: Attention weights, shape (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        tokens: Optional list of token strings for labels
        title: Title for visualization
    
    Returns:
        Dictionary with visualization metadata
    """
    if len(attention_weights.shape) == 3:
        # Multiple heads
        num_heads = attention_weights.shape[0]
        seq_len = attention_weights.shape[1]
    else:
        num_heads = 1
        seq_len = attention_weights.shape[0]
    
    if tokens is None:
        tokens = [f"tok_{i}" for i in range(seq_len)]
    
    if len(tokens) != seq_len:
        raise ValueError(f"Number of tokens ({len(tokens)}) must match sequence length ({seq_len})")
    
    vis_data = {
        'shape': attention_weights.shape,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'tokens': tokens,
        'title': title,
        'mean_attention': float(np.mean(attention_weights)),
        'std_attention': float(np.std(attention_weights)),
        'max_attention': float(np.max(attention_weights)),
        'min_attention': float(np.min(attention_weights)),
    }
    
    # Per-head statistics if multiple heads
    if num_heads > 1:
        vis_data['head_entropies'] = [
            -float(np.sum(attention_weights[h] * np.log(np.maximum(attention_weights[h], 1e-10))))
            for h in range(num_heads)
        ]
    
    return vis_data


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    n_components: int = 2,
) -> Dict[str, Any]:
    """Prepare embedding visualization data (dimensionality reduction).
    
    Args:
        embeddings: Embeddings array, shape (num_samples, embedding_dim)
        labels: Optional cluster/class labels, shape (num_samples,)
        method: Reduction method ('pca', 'tsne', 'umap'), default 'tsne'
        n_components: Number of dimensions to reduce to (2 or 3)
    
    Returns:
        Dictionary with original and reduced embeddings
    """
    num_samples, embedding_dim = embeddings.shape
    
    if n_components not in [2, 3]:
        n_components = 2
    
    # Simple PCA implementation
    if method == "pca":
        # Center embeddings
        centered = embeddings - np.mean(embeddings, axis=0)
        
        # Compute covariance and eigenvectors
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx[:n_components]]
        
        reduced = centered @ eigenvectors
    
    elif method == "tsne":
        # Simplified t-SNE-like approach: use PCA + random projection
        centered = embeddings - np.mean(embeddings, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx[:min(50, n_components)]]
        
        pca_reduced = centered @ eigenvectors
        
        # Random projection to final dimensions (not true t-SNE, but efficient approximation)
        rng_projection = np.random.RandomState(42).randn(pca_reduced.shape[1], n_components)
        reduced = pca_reduced @ rng_projection
    
    else:  # Default to PCA
        centered = embeddings - np.mean(embeddings, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx[:n_components]]
        reduced = centered @ eigenvectors
    
    vis_data = {
        'embeddings': embeddings,
        'reduced': reduced,
        'method': method,
        'n_components': n_components,
        'num_samples': num_samples,
        'embedding_dim': embedding_dim,
    }
    
    if labels is not None:
        if len(labels) != num_samples:
            raise ValueError(f"Labels length ({len(labels)}) must match embeddings ({num_samples})")
        vis_data['labels'] = labels
        
        # Compute silhouette score
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(reduced, metric='euclidean'))
        distances = np.maximum(distances, 1e-10)
        
        silhouette_scores = []
        for i in range(num_samples):
            same_label = np.where(labels == labels[i])[0]
            other_label = np.where(labels != labels[i])[0]
            
            a = np.mean(distances[i, same_label]) if len(same_label) > 1 else 0
            b = np.mean(distances[i, other_label]) if len(other_label) > 0 else 0
            
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhouette_scores.append(s)
        
        vis_data['silhouette_score'] = float(np.mean(silhouette_scores))
    
    return vis_data


def plot_loss_curve(
    losses: List[float],
    smoothing_window: int = 1,
) -> Dict[str, Any]:
    """Prepare loss curve visualization data.
    
    Args:
        losses: List of loss values
        smoothing_window: Window size for smoothing (default 1 for no smoothing)
    
    Returns:
        Dictionary with loss values and smoothed curve
    """
    losses = np.array(losses)
    steps = np.arange(len(losses))
    
    # Compute smoothed loss using moving average
    if smoothing_window > 1:
        smoothed = np.convolve(losses, np.ones(smoothing_window) / smoothing_window, mode='valid')
        smoothed_steps = steps[:len(smoothed)]
    else:
        smoothed = losses
        smoothed_steps = steps
    
    vis_data = {
        'losses': losses.tolist(),
        'smoothed_losses': smoothed.tolist(),
        'steps': steps.tolist(),
        'smoothed_steps': smoothed_steps.tolist(),
        'min_loss': float(np.min(losses)),
        'max_loss': float(np.max(losses)),
        'final_loss': float(losses[-1]),
        'smoothing_window': smoothing_window,
    }
    
    return vis_data


def plot_gradient_flow(
    gradients: Dict[str, np.ndarray],
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze gradient flow through network layers.
    
    Args:
        gradients: Dictionary of {layer_name: gradient_array}
        layer_names: Optional ordered list of layer names
    
    Returns:
        Dictionary with gradient statistics per layer
    """
    if layer_names is None:
        layer_names = sorted(gradients.keys())
    
    grad_stats = {}
    for name in layer_names:
        if name not in gradients:
            continue
        
        grad = gradients[name]
        grad_flat = grad.flatten()
        grad_flat = grad_flat[~np.isnan(grad_flat)]  # Remove NaNs
        grad_flat = grad_flat[~np.isinf(grad_flat)]  # Remove Infs
        
        grad_stats[name] = {
            'mean': float(np.mean(grad_flat)) if len(grad_flat) > 0 else 0.0,
            'std': float(np.std(grad_flat)) if len(grad_flat) > 0 else 0.0,
            'max': float(np.max(grad_flat)) if len(grad_flat) > 0 else 0.0,
            'min': float(np.min(grad_flat)) if len(grad_flat) > 0 else 0.0,
            'shape': grad.shape,
            'num_params': grad.size,
        }
    
    vis_data = {
        'layer_names': layer_names,
        'gradient_stats': grad_stats,
    }
    
    # Compute vanishing/exploding gradient indicators
    grad_magnitudes = [abs(stat['mean']) for stat in grad_stats.values() if 'mean' in stat]
    if grad_magnitudes:
        vis_data['mean_gradient_magnitude'] = float(np.mean(grad_magnitudes))
        vis_data['gradient_magnitude_ratio'] = float(np.max(grad_magnitudes) / (np.min(grad_magnitudes) + 1e-10))
    
    return vis_data


def plot_token_distribution(
    token_ids: np.ndarray,
    vocab_size: Optional[int] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Analyze token distribution in sequences.
    
    Args:
        token_ids: Array of token IDs
        vocab_size: Optional vocabulary size (for normalizing)
        top_k: Show top-k most frequent tokens (default 20)
    
    Returns:
        Dictionary with token frequency statistics
    """
    token_ids_flat = token_ids.flatten()
    
    unique, counts = np.unique(token_ids_flat, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    
    top_tokens = unique[sorted_idx[:top_k]]
    top_counts = counts[sorted_idx[:top_k]]
    
    if vocab_size is None:
        vocab_size = int(np.max(token_ids_flat)) + 1
    
    coverage = float(np.sum(top_counts) / len(token_ids_flat))
    
    vis_data = {
        'top_tokens': top_tokens.tolist(),
        'top_counts': top_counts.tolist(),
        'vocab_size': vocab_size,
        'unique_tokens': len(unique),
        'coverage': coverage,
        'total_tokens': len(token_ids_flat),
        'entropy': float(-np.sum((counts / len(token_ids_flat)) * np.log(np.maximum(counts / len(token_ids_flat), 1e-10)))),
    }
    
    return vis_data


def compare_attention_patterns(
    attention1: np.ndarray,
    attention2: np.ndarray,
) -> Dict[str, Any]:
    """Compare two attention weight matrices.
    
    Args:
        attention1: First attention matrix, shape (seq_len, seq_len)
        attention2: Second attention matrix, shape (seq_len, seq_len)
    
    Returns:
        Dictionary with comparison metrics
    """
    if attention1.shape != attention2.shape:
        raise ValueError("Attention matrices must have same shape")
    
    # Compute differences
    diff = attention1 - attention2
    
    comparison = {
        'shape': attention1.shape,
        'l1_difference': float(np.sum(np.abs(diff))),
        'l2_difference': float(np.sqrt(np.sum(diff ** 2))),
        'max_abs_diff': float(np.max(np.abs(diff))),
        'mean_abs_diff': float(np.mean(np.abs(diff))),
        'cosine_similarity': float(
            np.dot(attention1.flatten(), attention2.flatten()) / 
            (np.linalg.norm(attention1.flatten()) * np.linalg.norm(attention2.flatten()) + 1e-10)
        ),
    }
    
    return comparison
