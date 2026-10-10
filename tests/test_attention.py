import contextlib

import pytest
from utf8_tokenizer.control import ControlTokens

from welt.attention import get_shift_blocks

SO, SI = ControlTokens.ShiftOut, ControlTokens.ShiftIn


@pytest.mark.parametrize(("words", "warning", "blocks"), [
    (["a", "b", "c"], None, []),
    (["a", SO, "b", SI, "c"], None, [(1, 3)]),
    ([SO, SI, "c"], None, [(0, 1)]),
    (["a", SO, "b", SI, "c", SO, "d", SI], None, [(1, 3), (5, 7)]),
    (["a", SO, "b"], "Missing corresponding Shift In", []),
    (["a", SI, "b"], "Skipping self-attention block", []),
    (["a", SO, "b", SO, "c", SI], "Nested shift blocks are not allowed", [(3, 5)]),
])
def test_shift_blocks(words, warning, blocks):
    """Shift blocks span ShiftOut to ShiftIn, inclusive."""
    with pytest.warns(UserWarning, match=warning) if warning else contextlib.nullcontext():
        assert list(get_shift_blocks(words)) == blocks
