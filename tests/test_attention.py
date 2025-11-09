import pytest
import torch
from utf8_tokenizer.control import ControlTokens

from welt.attention import (
    add_self_attention_blocks,
    get_attention_mask_for_packed_sequence,
    get_position_ids_for_packed_sequence,
    get_shift_blocks,
)


def test_get_attention_mask_for_packed_sequence_single_sequence():
    seq_lengths = [3]
    mask = get_attention_mask_for_packed_sequence(seq_lengths)

    expected = torch.tensor([[
        [True, False, False],
        [True, True, False],
        [True, True, True]
    ]])

    assert torch.equal(mask, expected)
    assert mask.shape == (1, 3, 3)


def test_get_attention_mask_for_packed_sequence_two_sequences():
    seq_lengths = [2, 2]
    mask = get_attention_mask_for_packed_sequence(seq_lengths)

    expected = torch.tensor([[
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True]
    ]])

    assert torch.equal(mask, expected)
    assert mask.shape == (1, 4, 4)


def test_get_position_ids_for_packed_sequence_single_sequence():
    seq_lengths = [3]
    position_ids = get_position_ids_for_packed_sequence(seq_lengths)

    expected = torch.tensor([0, 1, 2])

    assert torch.equal(position_ids, expected)
    assert position_ids.shape == (3,)


def test_get_position_ids_for_packed_sequence_two_sequences():
    seq_lengths = [2, 2]
    position_ids = get_position_ids_for_packed_sequence(seq_lengths)

    expected = torch.tensor([0, 1, 0, 1])

    assert torch.equal(position_ids, expected)
    assert position_ids.shape == (4,)


def test_add_self_attention_blocks_basic_shift_block():
    mask = torch.zeros((1, 5, 5), dtype=torch.bool)
    words = ["hello", ControlTokens.ShiftOut, "world", ControlTokens.ShiftIn, "end"]

    add_self_attention_blocks(mask, words)

    expected = torch.zeros((1, 5, 5), dtype=torch.bool)
    expected[0, 1:4, 1:4] = True

    assert torch.equal(mask, expected)


def test_add_self_attention_blocks_multiple_shift_blocks():
    mask = torch.zeros((1, 8, 8), dtype=torch.bool)
    words = [
        "start",
        ControlTokens.ShiftOut, "first", "block", ControlTokens.ShiftIn,
        "middle",
        ControlTokens.ShiftOut, "second", ControlTokens.ShiftIn
    ]

    add_self_attention_blocks(mask, words)

    expected = torch.zeros((1, 8, 8), dtype=torch.bool)
    expected[0, 1:5, 1:5] = True
    expected[0, 6:8, 6:8] = True

    assert torch.equal(mask, expected)


def test_add_self_attention_blocks_no_shift_tokens():
    mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    words = ["hello", "world", "test"]

    original_mask = mask.clone()
    add_self_attention_blocks(mask, words)

    assert torch.equal(mask, original_mask)


def test_add_self_attention_blocks_only_shift_out():
    mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    words = ["hello", ControlTokens.ShiftOut, "world"]

    original_mask = mask.clone()
    with pytest.warns(UserWarning, match="Missing corresponding Shift In"):
        add_self_attention_blocks(mask, words)

    assert torch.equal(mask, original_mask)


def test_add_self_attention_blocks_shift_in_without_out():
    mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    words = ["hello", ControlTokens.ShiftIn, "world"]

    original_mask = mask.clone()

    with pytest.warns(UserWarning, match="Skipping self-attention block."):
        add_self_attention_blocks(mask, words)

    assert torch.equal(mask, original_mask)


def test_add_self_attention_blocks_single_token_block():
    mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    words = [ControlTokens.ShiftOut, ControlTokens.ShiftIn, "end"]

    add_self_attention_blocks(mask, words)

    expected = torch.zeros((1, 3, 3), dtype=torch.bool)
    expected[0, 0:2, 0:2] = True

    assert torch.equal(mask, expected)


def test_add_self_attention_blocks_nested_shift_out():
    mask = torch.zeros((1, 6, 6), dtype=torch.bool)
    words = [
        "start",
        ControlTokens.ShiftOut, "outer",
        ControlTokens.ShiftOut, "inner",
        ControlTokens.ShiftIn
    ]

    with pytest.warns(UserWarning, match="Nested shift blocks are not allowed."):
        add_self_attention_blocks(mask, words)

    expected = torch.zeros((1, 6, 6), dtype=torch.bool)
    expected[0, 3:6, 3:6] = True

    assert torch.equal(mask, expected)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
