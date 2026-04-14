"""Weight initialization strategies for neural networks."""

import numpy as np
from typing import Tuple


def xavier_uniform(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """Xavier/Glorot uniform initialization.
    
    Initializes weights uniformly within [-limit, limit] where
    limit = sqrt(6 / (fan_in + fan_out)) * gain
    
    Args:
        shape: Weight shape (typically (fan_in, fan_out))
        gain: Scaling factor (default 1.0)
    
    Returns:
        Initialized weight array
    """
    fan_in = shape[0] if len(shape) > 0 else 1
    fan_out = shape[1] if len(shape) > 1 else 1
    
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def xavier_normal(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """Xavier/Glorot normal initialization.
    
    Initializes weights from normal distribution with
    std = sqrt(2 / (fan_in + fan_out)) * gain
    
    Args:
        shape: Weight shape (typically (fan_in, fan_out))
        gain: Scaling factor (default 1.0)
    
    Returns:
        Initialized weight array
    """
    fan_in = shape[0] if len(shape) > 0 else 1
    fan_out = shape[1] if len(shape) > 1 else 1
    
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, size=shape)


def he_uniform(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """He/Kaiming uniform initialization (for ReLU).
    
    Initializes weights uniformly within [-limit, limit] where
    limit = sqrt(6 / fan_in) * gain
    
    Args:
        shape: Weight shape (typically (fan_in, fan_out))
        gain: Scaling factor (default 1.0; ~sqrt(2) for ReLU)
    
    Returns:
        Initialized weight array
    """
    fan_in = shape[0] if len(shape) > 0 else 1
    
    limit = gain * np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


def he_normal(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """He/Kaiming normal initialization (for ReLU).
    
    Initializes weights from normal distribution with
    std = sqrt(2 / fan_in) * gain
    
    Args:
        shape: Weight shape (typically (fan_in, fan_out))
        gain: Scaling factor (default 1.0; ~sqrt(2) for ReLU)
    
    Returns:
        Initialized weight array
    """
    fan_in = shape[0] if len(shape) > 0 else 1
    
    std = gain * np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, size=shape)


def normal(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """Normal distribution initialization.
    
    Args:
        shape: Weight shape
        mean: Mean of distribution (default 0.0)
        std: Standard deviation (default 1.0)
    
    Returns:
        Initialized weight array
    """
    return np.random.normal(mean, std, size=shape)


def uniform(shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """Uniform distribution initialization.
    
    Args:
        shape: Weight shape
        low: Lower bound (default -1.0)
        high: Upper bound (default 1.0)
    
    Returns:
        Initialized weight array
    """
    return np.random.uniform(low, high, size=shape)


def zeros(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize all zeros.
    
    Args:
        shape: Weight shape
    
    Returns:
        Zero-initialized array
    """
    return np.zeros(shape)


def ones(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize all ones.
    
    Args:
        shape: Weight shape
    
    Returns:
        One-initialized array
    """
    return np.ones(shape)


def orthogonal(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """Orthogonal matrix initialization (for square or rectangular matrices).
    
    Uses QR decomposition to create an orthogonal matrix.
    
    Args:
        shape: Weight shape (must have at least 2 dimensions)
        gain: Scaling factor (default 1.0)
    
    Returns:
        Orthogonal initialized array
    """
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2D shape")
    
    # Create random matrix and perform QR decomposition
    random_mat = np.random.normal(0, 1, size=shape)
    q, r = np.linalg.qr(random_mat.reshape(-1, shape[-1]))
    
    # Ensure positive diagonal
    d = np.diag(np.sign(np.diag(r)))
    q = q @ d
    
    return gain * q.reshape(shape)
