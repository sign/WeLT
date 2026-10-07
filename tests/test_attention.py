import contextlib

import pytest
import torch
from utf8_tokenizer.control import ControlTokens

from welt.attention import get_attention_mask_for_packed_sequence, get_shift_blocks


def test_get_attention_mask_for_packed_sequence_single_sequence():
    seq_lengths = [3]
    mask = get_attention_mask_for_packed_sequence(seq_lengths, ["x"] * sum(seq_lengths))

    expected = torch.tensor([[
        [True, False, False],
        [True, True, False],
        [True, True, True]
    ]])

    assert torch.equal(mask, expected)
    assert mask.shape == (1, 3, 3)


def test_get_attention_mask_for_packed_sequence_two_sequences():
    seq_lengths = [2, 2]
    mask = get_attention_mask_for_packed_sequence(seq_lengths, ["x"] * sum(seq_lengths))

    expected = torch.tensor([[
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True]
    ]])

    assert torch.equal(mask, expected)
    assert mask.shape == (1, 4, 4)


@pytest.mark.parametrize(("words", "warning", "blocks"), [
    (["a", ControlTokens.ShiftOut, "b", ControlTokens.ShiftIn, "c"], None, [(1, 4)]),
    ([ControlTokens.ShiftOut, ControlTokens.ShiftIn, "c"], None, [(0, 2)]),
    (["a", ControlTokens.ShiftOut, "b"], "Missing corresponding Shift In", []),
    (["a", ControlTokens.ShiftIn, "b"], "Skipping self-attention block", []),
    (["a", ControlTokens.ShiftOut, "b", ControlTokens.ShiftOut, "c", ControlTokens.ShiftIn],
     "Nested shift blocks are not allowed", [(3, 6)]),
])
def test_shift_blocks_are_bidirectional(words, warning, blocks):
    with pytest.warns(UserWarning, match=warning) if warning else contextlib.nullcontext():
        mask = get_attention_mask_for_packed_sequence([len(words)], words)
    expected = torch.ones(len(words), len(words), dtype=torch.bool).tril()
    for start, end in blocks:
        expected[start:end, start:end] = True
    assert torch.equal(mask[0], expected)


def test_get_attention_mask_for_packed_sequence_with_shift_blocks():
    seq_lengths = [7]
    words = [
        "hello",
        ControlTokens.ShiftOut, "prefix", "block", ControlTokens.ShiftIn,
        "world", "end"
    ]

    mask = get_attention_mask_for_packed_sequence(seq_lengths, words)

    expected = torch.tensor([[
        [True, False, False, False, False, False, False],
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, True, False],
        [True, True, True, True, True, True, True]
    ]])

    assert torch.equal(mask, expected)


def test_get_shift_blocks_single_block():
    """Test get_shift_blocks returns correct indexes for a single shift block."""
    words = ["hello", ControlTokens.ShiftOut, "world", "test", ControlTokens.ShiftIn, "end"]
    blocks = list(get_shift_blocks(words))

    assert len(blocks) == 1
    assert blocks[0] == (1, 4)  # ShiftOut at index 1, ShiftIn at index 4


def test_get_shift_blocks_multiple_blocks():
    """Test get_shift_blocks returns correct indexes for multiple shift blocks."""
    words = [
        "start",
        ControlTokens.ShiftOut, "first", ControlTokens.ShiftIn,
        "middle",
        ControlTokens.ShiftOut, "second", ControlTokens.ShiftIn,
        "end"
    ]
    blocks = list(get_shift_blocks(words))

    assert len(blocks) == 2
    assert blocks[0] == (1, 3)  # First block: ShiftOut at 1, ShiftIn at 3
    assert blocks[1] == (5, 7)  # Second block: ShiftOut at 5, ShiftIn at 7


def test_get_shift_blocks_no_blocks():
    """Test get_shift_blocks returns empty when no shift blocks present."""
    words = ["hello", "world", "test"]
    blocks = list(get_shift_blocks(words))

    assert len(blocks) == 0
