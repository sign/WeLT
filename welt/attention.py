import warnings

import torch
from utf8_tokenizer.control import ControlTokens

# Module-level caches that grow as needed
_tril_cache: torch.Tensor | None = None
_arange_cache: torch.Tensor | None = None


def _get_tril(size: int) -> torch.Tensor:
    """Get a lower triangular matrix of at least the given size, using cached version if possible."""
    global _tril_cache
    if _tril_cache is None or len(_tril_cache) < size:
        _tril_cache = torch.tril(torch.ones((size, size), dtype=torch.bool))
    return _tril_cache


def _get_arange(size: int) -> torch.Tensor:
    """Get an arange tensor of at least the given size, using cached version if possible."""
    global _arange_cache
    if _arange_cache is None or len(_arange_cache) < size:
        _arange_cache = torch.arange(size, dtype=torch.long)
    return _arange_cache


def get_shift_blocks(words: list[str]):
    """
    Find shift blocks in a sequence of words.

    Yields tuples (start, end) where start is the index of ShiftOut
    and end is the index of ShiftIn (inclusive). Handles warnings for invalid blocks.

    Args:
        words: List of word strings

    Yields:
        Tuples of (start_idx, end_idx) for each valid shift block
    """
    shift_out_idx = None

    for i, word in enumerate(words):
        if word == ControlTokens.ShiftOut:
            if shift_out_idx is not None:
                warnings.warn(
                    "Shift Out (SO) detected after another Shift Out (SO) without Shift In (SI). "
                    "Nested shift blocks are not allowed.",
                    stacklevel=2)
            shift_out_idx = i
        if word == ControlTokens.ShiftIn:
            if shift_out_idx is None:
                warnings.warn(
                    "Shift In (SI) detected, without first seeing Shift Out (SO). "
                    "Skipping self-attention block.",
                    stacklevel=2)
            else:
                yield shift_out_idx, i
                shift_out_idx = None

    if shift_out_idx is not None:
        warnings.warn(
            "Unclosed Shift Out (SO) block detected at end of sequence. "
            "Missing corresponding Shift In (SI).",
            stacklevel=2)


def add_self_attention_blocks(mask: torch.Tensor, words: list[str]) -> None:
    # Attention blocks (PrefixLM / MAS) are surrounded by <ShiftOut> and <ShiftIn> tokens (`\xOE` ... `\x0F`).
    for start, end in get_shift_blocks(words):
        mask[0, start:end + 1, start:end + 1] = 1


def get_attention_mask_for_packed_sequence(seq_lengths: list[int], words: list[str] = None) -> torch.Tensor:
    """
    Returns a 3D attention mask for a packed sequence. (1, seq_len, seq_len)
    The first dimension represents the head dimension, which is set to 1 for broadcasting.
    """
    total_length = sum(seq_lengths)

    mask = torch.zeros((1, total_length, total_length), dtype=torch.bool)

    # Use module-level cached tril matrix
    max_len = max(seq_lengths)
    tril = _get_tril(max_len)

    start_position = 0
    for length in seq_lengths:
        end_position = start_position + length
        mask[0, start_position:end_position, start_position:end_position] = tril[:length, :length]
        start_position = end_position

    if words is not None:
        add_self_attention_blocks(mask, words)

    return mask


def get_position_ids_for_packed_sequence(seq_lengths: list[int]) -> torch.Tensor:
    # Use module-level cached arange and slice
    max_len = max(seq_lengths)
    arange = _get_arange(max_len)
    return torch.cat([arange[:length] for length in seq_lengths])
