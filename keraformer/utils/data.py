"""Data loading and preprocessing utilities."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple, List, Union


@dataclass
class Dataset:
    """Simple dataset wrapper for sequences."""
    sequences: np.ndarray  # Shape (num_samples, seq_length) or (num_samples, seq_length, feature_dim)
    labels: Optional[np.ndarray] = None  # Shape (num_samples,) or (num_samples, output_dim)
    padding_mask: Optional[np.ndarray] = None  # Shape (num_samples, seq_length) with 1 for valid, 0 for padded
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Union[np.ndarray, Tuple]:
        if self.labels is None:
            if self.padding_mask is None:
                return self.sequences[idx]
            else:
                return self.sequences[idx], self.padding_mask[idx]
        else:
            if self.padding_mask is None:
                return self.sequences[idx], self.labels[idx]
            else:
                return self.sequences[idx], self.labels[idx], self.padding_mask[idx]


class DataLoader:
    """Simple data loader for batching sequences."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        pad_token_id: int = 0,
    ):
        """Initialize data loader.
        
        Args:
            dataset: Dataset object with sequences and optional labels/masks
            batch_size: Batch size (default 32)
            shuffle: Whether to shuffle data (default False)
            pad_token_id: Token ID for padding (default 0)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_token_id = pad_token_id
        self.indices = np.arange(len(dataset))
    
    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        indices = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            # Get batch data
            batch_sequences = self.dataset.sequences[batch_indices]
            batch = [batch_sequences]
            
            if self.dataset.labels is not None:
                batch.append(self.dataset.labels[batch_indices])
            
            if self.dataset.padding_mask is not None:
                batch.append(self.dataset.padding_mask[batch_indices])
            
            # Pad to same length within batch if needed
            max_len = np.max([seq.shape[0] for seq in batch_sequences])
            padded_batch = pad_sequences(batch_sequences, max_len, self.pad_token_id)
            
            if len(batch) == 1:
                yield padded_batch
            elif len(batch) == 2:
                yield padded_batch, batch[1]
            else:
                yield padded_batch, batch[1], batch[2]
    
    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def pad_sequences(
    sequences: Union[List[np.ndarray], np.ndarray],
    max_length: Optional[int] = None,
    pad_value: float = 0.0,
    pad_side: str = "right",
) -> np.ndarray:
    """Pad sequences to same length.
    
    Args:
        sequences: List of arrays or array of shape (batch, seq_len, ...)
        max_length: Maximum length to pad to (default: max of input sequences)
        pad_value: Value to pad with (default 0.0)
        pad_side: Which side to pad ("right" or "left", default "right")
    
    Returns:
        Padded array of shape (batch, max_length, ...)
    """
    if isinstance(sequences, np.ndarray):
        if len(sequences.shape) == 1:
            sequences = [sequences]
        else:
            # Already array, get max length from sequences
            if max_length is None:
                max_length = np.max([s.shape[0] for s in sequences])
            
            padded = []
            for seq in sequences:
                if seq.shape[0] < max_length:
                    pad_width = [(0, max_length - seq.shape[0])] + [(0, 0)] * (len(seq.shape) - 1)
                    if pad_side == "left":
                        pad_width = [(max_length - seq.shape[0], 0)] + [(0, 0)] * (len(seq.shape) - 1)
                    padded.append(np.pad(seq, pad_width, constant_values=pad_value))
                else:
                    padded.append(seq[:max_length])
            
            return np.array(padded)
    
    # List of sequences
    if max_length is None:
        max_length = max(len(s) for s in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            if pad_side == "right":
                pad_width = (0, max_length - len(seq))
            else:
                pad_width = (max_length - len(seq), 0)
            
            if isinstance(seq, np.ndarray):
                padded_seq = np.pad(seq, pad_width, constant_values=pad_value)
            else:
                padded_seq = np.pad(np.array(seq), pad_width, constant_values=pad_value)
            
            padded.append(padded_seq)
        else:
            padded.append(np.array(seq)[:max_length])
    
    return np.array(padded)


def create_batches(
    sequences: np.ndarray,
    labels: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    pad_token_id: int = 0,
) -> Iterator[Tuple]:
    """Create batches from sequences with optional labels.
    
    Args:
        sequences: Array of sequences, shape (num_samples, seq_length, ...)
        labels: Optional labels array, shape (num_samples, ...)
        batch_size: Batch size (default 32)
        shuffle: Whether to shuffle batches (default False)
        pad_token_id: Padding token ID (default 0)
    
    Yields:
        Tuples of (batch_sequences, batch_labels) or just batch_sequences
    """
    num_samples = len(sequences)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_seqs = sequences[batch_indices]
        
        # Pad to max length in batch
        if len(batch_seqs.shape) > 1:
            max_len = np.max([s.shape[0] for s in batch_seqs])
            batch_seqs = pad_sequences(batch_seqs, max_len, pad_token_id)
        
        if labels is not None:
            batch_labels = labels[batch_indices]
            yield batch_seqs, batch_labels
        else:
            yield batch_seqs


def create_autoregressive_dataset(
    sequences: np.ndarray,
    target_length: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create autoregressive input-target pairs from sequences.
    
    Shifts sequences to create (input, target) pairs for next-token prediction.
    
    Args:
        sequences: Array of shape (num_samples, seq_length)
        target_length: How many tokens to shift for target
        stride: Stride for sliding window (default 1)
    
    Returns:
        Tuple of (inputs, targets) each of shape (num_pairs,) or (num_pairs, length)
    """
    inputs = []
    targets = []
    
    for seq in sequences:
        for i in range(0, len(seq) - target_length, stride):
            inputs.append(seq[i:-target_length])
            targets.append(seq[i+target_length])

    # Handle variable-length inputs by using object arrays if needed.
    try:
        inputs_array = np.array(inputs)
    except ValueError:
        inputs_array = np.empty(len(inputs), dtype=object)
        for idx, inp in enumerate(inputs):
            inputs_array[idx] = inp

    targets_array = np.array(targets)
    return inputs_array, targets_array


def create_mask_for_padding(
    sequences: np.ndarray,
    pad_token_id: int = 0,
) -> np.ndarray:
    """Create binary padding mask from sequences.
    
    Args:
        sequences: Array of sequence IDs, shape (batch, seq_length)
        pad_token_id: Token ID representing padding (default 0)
    
    Returns:
        Binary mask of shape (batch, seq_length) with 1 for valid tokens, 0 for padding
    """
    return (sequences != pad_token_id).astype(np.float32)


def create_causal_mask(seq_length: int) -> np.ndarray:
    """Create causal mask for autoregressive attention.
    
    Args:
        seq_length: Length of sequence
    
    Returns:
        Causal mask of shape (seq_length, seq_length) with 1 for visible, 0 for masked
    """
    mask = np.tril(np.ones((seq_length, seq_length), dtype=np.float32))
    return mask


def batch_size_aware_pack(
    sequences: np.ndarray,
    target_batch_tokens: int = 4096,
    pad_token_id: int = 0,
) -> List[np.ndarray]:
    """Pack sequences into batches based on token count.
    
    Groups sequences to approximately match target total tokens per batch.
    
    Args:
        sequences: List of sequence arrays with varying lengths
        target_batch_tokens: Target number of tokens per batch (default 4096)
        pad_token_id: Token for padding (default 0)
    
    Returns:
        List of batched arrays
    """
    if isinstance(sequences, np.ndarray) and len(sequences.shape) == 1:
        sequences = [sequences]
    
    # Sort by length (descending) for better packing
    sorted_indices = np.argsort([len(s) for s in sequences])[::-1]
    sorted_seqs = [sequences[i] for i in sorted_indices]
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for seq in sorted_seqs:
        seq_len = len(seq)
        batch_size = len(current_batch) + 1
        total_tokens = seq_len * batch_size + sum(len(s) for s in current_batch)
        
        if total_tokens <= target_batch_tokens or not current_batch:
            current_batch.append(seq)
            current_tokens = total_tokens
        else:
            if current_batch:
                batches.append(pad_sequences(current_batch, pad_value=pad_token_id))
            current_batch = [seq]
            current_tokens = seq_len
    
    if current_batch:
        batches.append(pad_sequences(current_batch, pad_value=pad_token_id))
    
    return batches
