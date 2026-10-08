import random

import pytest

pytest.importorskip("vllm", reason="Requires the NeMo container")

from welt.utf8 import utf8_transitions  # noqa: E402

TRANSITIONS = utf8_transitions().tolist()


def accepts(data: bytes) -> tuple[bool, bool]:
    """(every byte is allowed, ends at a character boundary)"""
    state = 0
    for byte in data:
        state = TRANSITIONS[state][byte]
        if state < 0:
            return False, False
    return True, state == 0


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_accepts_exactly_valid_utf8(length):
    rng = random.Random(length)
    candidates = [bytes(rng.choice([rng.randrange(256), rng.randrange(0x80, 0xC0), rng.randrange(0xC0, 0xF8)])
                        for _ in range(length)) for _ in range(20_000)]
    for data in candidates:
        allowed, complete = accepts(data)
        try:
            data.decode("utf-8")
            valid = True
        except UnicodeDecodeError:
            valid = False
        assert complete == valid, data  # Whole sequences: accepted at a boundary iff valid UTF-8
        if allowed and not complete:  # An allowed incomplete sequence can be completed into valid UTF-8
            assert any(accepts(data + tail)[1] for tail in completions(data)), data


def completions(data: bytes):
    """Continuation byte sequences (1-3 bytes) that could complete the last character."""
    for first in range(0x80, 0xC0):
        yield bytes([first])
        for second in (0x80, 0xBF):
            yield bytes([first, second])
            yield bytes([first, second, 0x80])


def test_text_is_accepted():
    assert accepts("héllo שלום 👋 日本".encode()) == (True, True)
