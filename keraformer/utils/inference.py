"""Inference utilities for text generation and model interaction."""

import numpy as np
from typing import Tuple


def greedy_decode(
    logits: np.ndarray,
    sequence_length: int,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
    max_length: int = 512,
    temperature: float = 1.0,
) -> np.ndarray:
    """Greedy decoding: select highest probability token at each step.
    
    Args:
        logits: Model logits of shape (batch, vocab_size) or (seq_len, vocab_size)
        sequence_length: Current sequence length
        pad_token_id: Padding token ID (default 0)
        eos_token_id: End-of-sequence token ID (default 2)
        max_length: Maximum generation length (default 512)
        temperature: Softmax temperature for scaling logits (default 1.0)
    
    Returns:
        Generated token ids of shape (max_length,) or (batch, max_length)
    """
    batch_size = logits.shape[0] if len(logits.shape) > 1 else 1
    
    # Initialize output sequence
    is_batched = len(logits.shape) > 1
    if is_batched:
        generated = np.full((batch_size, max_length), pad_token_id, dtype=np.int32)
        generated[:, :sequence_length] = pad_token_id  # placeholder
    else:
        generated = np.full((max_length,), pad_token_id, dtype=np.int32)
        generated[:sequence_length] = pad_token_id  # placeholder
    
    # Apply temperature scaling and select argmax
    scaled_logits = logits / (temperature + 1e-10)
    next_token = np.argmax(scaled_logits, axis=-1)
    
    return next_token


def beam_search(
    logits: np.ndarray,
    beam_width: int = 3,
    max_length: int = 512,
    length_penalty: float = 0.0,
    eos_token_id: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Beam search decoding.
    
    Args:
        logits: Model logits of shape (vocab_size,) or (batch, vocab_size)
        beam_width: Number of beams (default 3)
        max_length: Maximum generation length (default 512)
        length_penalty: Length penalty alpha (default 0.0 for no penalty)
        eos_token_id: End-of-sequence token ID (default 2)
    
    Returns:
        Tuple of (beam_sequences, beam_scores):
        - beam_sequences: shape (beam_width, max_length) top-k sequences
        - beam_scores: shape (beam_width,) normalized scores
    """
    # Get top-k tokens
    top_logits, top_indices = np.sort(logits)[::-1][:beam_width], np.argsort(logits)[::-1][:beam_width]
    
    beam_sequences = np.expand_dims(top_indices, axis=1)  # (beam_width, 1)
    beam_scores = top_logits  # (beam_width,)
    
    # Length-normalize scores
    if length_penalty > 0:
        beam_scores = beam_scores / ((beam_width ** length_penalty))
    
    return beam_sequences, beam_scores


def temperature_sampling(
    logits: np.ndarray,
    temperature: float = 1.0,
    num_samples: int = 1,
) -> np.ndarray:
    """Temperature-based sampling: smooth probability distribution.
    
    Higher temperature → flatter distribution (more diversity)
    Lower temperature → sharper distribution (more confident)
    
    Args:
        logits: Model logits of shape (vocab_size,) or (batch, vocab_size)
        temperature: Temperature scaling (default 1.0)
        num_samples: Number of samples to draw (default 1)
    
    Returns:
        Sampled token ids of shape (num_samples,) or (batch, num_samples)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(scaled_logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Sample tokens
    is_batched = len(logits.shape) > 1
    if is_batched:
        batch_size = logits.shape[0]
        samples = np.array([
            np.random.choice(logits.shape[-1], size=num_samples, p=probs[i])
            for i in range(batch_size)
        ])
    else:
        samples = np.random.choice(logits.shape[-1], size=num_samples, p=probs)
    
    return samples


def top_k_sampling(
    logits: np.ndarray,
    k: int = 10,
    temperature: float = 1.0,
) -> np.ndarray:
    """Top-k filtering: keep only top-k most probable tokens.
    
    Args:
        logits: Model logits of shape (vocab_size,) or (batch, vocab_size)
        k: Number of top tokens to keep (default 10)
        temperature: Temperature scaling (default 1.0)
    
    Returns:
        Sampled token ids
    """
    scaled_logits = logits / temperature
    
    # Get top-k
    is_batched = len(logits.shape) > 1
    if is_batched:
        top_k_indices = np.argsort(scaled_logits)[:, -k:]
        # Create mask and set other logits to -inf
        mask = np.full_like(scaled_logits, -np.inf)
        batch_size = scaled_logits.shape[0]
        for i in range(batch_size):
            mask[i, top_k_indices[i, :]] = scaled_logits[i, top_k_indices[i, :]]
        filtered_logits = mask
    else:
        top_k_indices = np.argsort(scaled_logits)[-k:]
        mask = np.full_like(scaled_logits, -np.inf)
        mask[top_k_indices] = scaled_logits[top_k_indices]
        filtered_logits = mask
    
    return temperature_sampling(filtered_logits, temperature=temperature)


def top_p_sampling(
    logits: np.ndarray,
    p: float = 0.9,
    temperature: float = 1.0,
) -> np.ndarray:
    """Nucleus/Top-p sampling: keep tokens with cumulative probability >= p.
    
    Args:
        logits: Model logits of shape (vocab_size,) or (batch, vocab_size)
        p: Cumulative probability threshold (default 0.9)
        temperature: Temperature scaling (default 1.0)
    
    Returns:
        Sampled token ids
    """
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    
    scaled_logits = logits / temperature
    
    is_batched = len(logits.shape) > 1
    if is_batched:
        batch_size = scaled_logits.shape[0]
        samples = []
        for i in range(batch_size):
            # Sort by logits descending
            sorted_indices = np.argsort(scaled_logits[i])[::-1]
            sorted_logits = scaled_logits[i, sorted_indices]
            
            # Compute cumulative probabilities
            sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
            sorted_probs = sorted_probs / np.sum(sorted_probs)
            cumsum_probs = np.cumsum(sorted_probs)
            
            # Mask tokens outside cumulative probability
            sorted_indices_mask = sorted_indices[cumsum_probs <= p]
            if len(sorted_indices_mask) == 0:  # fallback: keep at least top token
                sorted_indices_mask = sorted_indices[:1]
            
            # Sample from filtered vocabulary
            filtered_logits = np.full_like(scaled_logits[i], -np.inf)
            filtered_logits[sorted_indices_mask] = scaled_logits[i, sorted_indices_mask]
            
            # Temperature sampling on filtered distribution
            sample = temperature_sampling(filtered_logits, temperature=1.0, num_samples=1)
            samples.append(sample[0])
        
        return np.array(samples)
    else:
        sorted_indices = np.argsort(scaled_logits)[::-1]
        sorted_logits = scaled_logits[sorted_indices]
        
        sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
        sorted_probs = sorted_probs / np.sum(sorted_probs)
        cumsum_probs = np.cumsum(sorted_probs)
        
        sorted_indices_mask = sorted_indices[cumsum_probs <= p]
        if len(sorted_indices_mask) == 0:
            sorted_indices_mask = sorted_indices[:1]
        
        filtered_logits = np.full_like(scaled_logits, -np.inf)
        filtered_logits[sorted_indices_mask] = scaled_logits[sorted_indices_mask]
        
        return temperature_sampling(filtered_logits, temperature=1.0, num_samples=1)
