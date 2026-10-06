"""Metric utilities for WeLT training."""

import json
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


def count_utf8_bytes(tokenizer, token_ids) -> int:
    """Count text bytes without turning partial ByteLevel UTF-8 into replacements."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if decoder is not None and json.loads(decoder.__getstate__()).get("type") == "ByteLevel":
        # ByteLevel maps each original byte to one Unicode alphabet symbol.
        # Decoding a chunk through text first can inflate a split multibyte
        # character into a three-byte replacement character at chunk boundaries.
        added_ids = set(tokenizer.get_added_vocab().values())
        tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
        return sum(len(token.encode("utf-8")) if int(token_id) in added_ids else len(token)
                   for token_id, token in zip(token_ids, tokens, strict=True))
    text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return len(text.encode("utf-8"))
