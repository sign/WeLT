import contextlib

import pytest
import torch
from utf8_tokenizer.control import ControlTokens

from welt.attention import get_attention_mask_for_packed_sequence, get_shift_blocks

SO, SI = ControlTokens.ShiftOut, ControlTokens.ShiftIn


@pytest.mark.parametrize(("seq_lengths", "words", "warning", "blocks"), [
    ([3], ["a", "b", "c"], None, []),
    ([2, 2], ["a", "b", "c", "d"], None, []),  # Packed sequences do not attend to each other
    ([5], ["a", SO, "b", SI, "c"], None, [(1, 3)]),
    ([3], [SO, SI, "c"], None, [(0, 1)]),
    ([8], ["a", SO, "b", SI, "c", SO, "d", SI], None, [(1, 3), (5, 7)]),
    ([3], ["a", SO, "b"], "Missing corresponding Shift In", []),
    ([3], ["a", SI, "b"], "Skipping self-attention block", []),
    ([6], ["a", SO, "b", SO, "c", SI], "Nested shift blocks are not allowed", [(3, 5)]),
])
def test_attention_mask(seq_lengths, words, warning, blocks):
    """Causal within each packed sequence, bidirectional within shift blocks (ShiftOut to ShiftIn, inclusive)."""
    with pytest.warns(UserWarning, match=warning) if warning else contextlib.nullcontext():
        assert list(get_shift_blocks(words)) == blocks
        mask = get_attention_mask_for_packed_sequence(seq_lengths, words)
    expected = torch.block_diag(*[torch.ones(n, n, dtype=torch.bool).tril() for n in seq_lengths])
    for start, end in blocks:
        expected[start:end + 1, start:end + 1] = True
    assert torch.equal(mask, expected[None])
