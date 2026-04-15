"""Model checkpointing utilities for saving and loading weights."""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    path: str,
    weights: Dict[str, np.ndarray],
    optimizer_state: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    step: int = 0,
) -> None:
    """Save model weights, optimizer state, and metadata to disk.
    
    Args:
        path: Path to save checkpoint (e.g., 'checkpoints/model_step100.npz')
        weights: Dictionary of weight arrays {name: array}
        optimizer_state: Optional optimizer state dict (e.g., adam_m, adam_v)
        metadata: Optional metadata dict (e.g., config, epoch, loss)
        step: Training step/epoch number (default 0)
    
    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare all data to save
    save_dict = dict(weights)
    
    if optimizer_state is not None:
        for key, val in optimizer_state.items():
            save_dict[f"__optimizer__{key}"] = val
    
    if metadata is not None:
        for key, val in metadata.items():
            if isinstance(val, (np.ndarray, dict, list)):
                save_dict[f"__metadata__{key}"] = val
            # Scalars handled separately
    
    # Add step and metadata scalars
    save_dict["__step__"] = np.array(step)
    if metadata is not None:
        for key, val in metadata.items():
            if isinstance(val, (int, float, str)):
                save_dict[f"__metadata__{key}"] = np.array(val)
    
    # Save as NPZ (compressed numpy zip)
    np.savez_compressed(path, **save_dict)


def load_checkpoint(
    path: str,
    return_optimizer_state: bool = True,
    return_metadata: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint from disk.
    
    Args:
        path: Path to checkpoint file
        return_optimizer_state: Include optimizer state in output (default True)
        return_metadata: Include metadata in output (default True)
    
    Returns:
        Dictionary with keys:
        - 'weights': dict of weight arrays
        - 'optimizer_state': dict of optimizer arrays (if return_optimizer_state=True)
        - 'metadata': dict of metadata (if return_metadata=True)
        - 'step': training step number
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    with np.load(path, allow_pickle=True) as data:
        checkpoint = {}
        weights = {}
        optimizer_state = {}
        metadata = {}
        step = 0
        
        for key, val in data.items():
            key_str = str(key)
            
            if key_str == "__step__":
                step = int(val)
            elif key_str.startswith("__optimizer__"):
                opt_key = key_str.replace("__optimizer__", "")
                optimizer_state[opt_key] = val
            elif key_str.startswith("__metadata__"):
                meta_key = key_str.replace("__metadata__", "")
                metadata[meta_key] = val
            else:
                weights[key_str] = val
        
        checkpoint['weights'] = weights
        checkpoint['step'] = step
        
        if return_optimizer_state and optimizer_state:
            checkpoint['optimizer_state'] = optimizer_state
        
        if return_metadata and metadata:
            checkpoint['metadata'] = metadata
        
        return checkpoint


def get_checkpoint_info(path: str) -> Dict[str, Any]:
    """Get information about checkpoint without loading full weights.
    
    Args:
        path: Path to checkpoint file
    
    Returns:
        Dictionary with:
        - 'step': training step
        - 'weight_names': list of weight array names
        - 'weight_shapes': list of weight shapes
        - 'optimizer_keys': list of optimizer state keys
        - 'metadata': metadata dict
        - 'file_size_mb': checkpoint file size in MB
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    with np.load(path, allow_pickle=True) as data:
        info = {
            'weight_names': [],
            'weight_shapes': [],
            'optimizer_keys': [],
            'metadata': {},
            'step': 0,
            'file_size_mb': os.path.getsize(path) / (1024 * 1024),
        }
        
        for key, val in data.items():
            key_str = str(key)
            
            if key_str == "__step__":
                info['step'] = int(val)
            elif key_str.startswith("__optimizer__"):
                opt_key = key_str.replace("__optimizer__", "")
                info['optimizer_keys'].append(opt_key)
            elif key_str.startswith("__metadata__"):
                meta_key = key_str.replace("__metadata__", "")
                info['metadata'][meta_key] = val
            else:
                info['weight_names'].append(key_str)
                info['weight_shapes'].append(val.shape)
        
        return info


def find_latest_checkpoint(directory: str) -> Optional[str]:
    """Find the latest checkpoint in a directory.
    
    Assumes checkpoint filenames containing step numbers (e.g., model_step100.npz)
    
    Args:
        directory: Directory to search for checkpoints
    
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(directory)
    if not checkpoint_dir.exists():
        return None
    
    npz_files = list(checkpoint_dir.glob("*.npz"))
    if not npz_files:
        return None
    
    # Sort by modification time
    latest = max(npz_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def compare_checkpoints(path1: str, path2: str, atol: float = 1e-5) -> Dict[str, Any]:
    """Compare two checkpoints for differences.
    
    Args:
        path1: Path to first checkpoint
        path2: Path to second checkpoint
        atol: Absolute tolerance for floating point comparison
    
    Returns:
        Dictionary with comparison results including:
        - 'step_diff': difference in training steps
        - 'same_weights': set of weight names that are identical
        - 'different_weights': dict mapping weight names to max absolute difference
        - 'only_in_path1': weights only in checkpoint 1
        - 'only_in_path2': weights only in checkpoint 2
    """
    ckpt1 = load_checkpoint(path1, return_optimizer_state=False, return_metadata=False)
    ckpt2 = load_checkpoint(path2, return_optimizer_state=False, return_metadata=False)
    
    weights1 = ckpt1['weights']
    weights2 = ckpt2['weights']
    
    comparison = {
        'step_diff': ckpt2['step'] - ckpt1['step'],
        'same_weights': set(),
        'different_weights': {},
        'only_in_path1': set(),
        'only_in_path2': set(),
    }
    
    # Check common weights
    for key in weights1.keys():
        if key not in weights2:
            comparison['only_in_path1'].add(key)
        else:
            max_diff = np.max(np.abs(weights1[key] - weights2[key]))
            if max_diff < atol:
                comparison['same_weights'].add(key)
            else:
                comparison['different_weights'][key] = float(max_diff)
    
    # Check weights only in path2
    for key in weights2.keys():
        if key not in weights1:
            comparison['only_in_path2'].add(key)
    
    return comparison
