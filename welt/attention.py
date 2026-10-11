import warnings

from utf8_tokenizer.control import ControlTokens


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
