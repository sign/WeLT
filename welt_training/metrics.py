"""Metric utilities for WeLT training."""

import math


def compute_bits_per_byte(loss: float, num_tokens: int, num_bytes: int) -> float:
    """
    Compute bits per byte (BPB) from average cross-entropy loss.

    Converts per-token cross-entropy loss (in nats) to bits per byte.
    For byte-level models where num_tokens == num_bytes, this simplifies
    to loss / ln(2).

    Args:
        loss: Average cross-entropy loss per token (in nats).
        num_tokens: Number of tokens the loss was averaged over.
        num_bytes: Total number of bytes in the original text.

    Returns:
        Bits per byte.
    """
    if num_bytes == 0:
        return float("inf")
    total_bits = loss * num_tokens / math.log(2)
    return total_bits / num_bytes
