import pytest
import torch

pytest.importorskip("vllm")

from welt import ContextTooLongError  # noqa: E402
from welt.inference import WeLTGenerator  # noqa: E402
from welt.processor import TextImageProcessor  # noqa: E402


def stub_generator(next_words, max_model_len=100):
    """A WeLTGenerator without engines: _next_words(prefixes) returns each active text's next word (bytes)."""
    generator = object.__new__(WeLTGenerator)
    generator.processor = TextImageProcessor.create(max_word_length=32, render_images=False)
    model_config = type("ModelConfig", (), {"max_model_len": max_model_len})
    generator.latent = type("LLM", (), {"llm_engine": type("Engine", (), {"model_config": model_config})})
    generator._latents = lambda sequences: torch.zeros(len(sequences), 1)
    calls = []

    def _next_words(latents, prefixes, sampling):
        calls.append((list(prefixes), [params.seed for params in sampling]))
        return [next_words(prefix) for prefix in prefixes]

    generator._next_words = _next_words
    return generator, calls


def test_generation_continues_a_partial_word_and_stops_on_an_empty_word():
    words = iter([b"world ", b""])
    generator, calls = stub_generator(lambda prefix: prefix + b"ld " if prefix else next(words))
    assert generator.generate_words(["hello wor"]) == [["ld ", "world "]]
    assert [prefixes for prefixes, _ in calls] == [[b"wor"], [b""], [b""]]


def test_generation_stops_at_max_generated_words_with_a_seed_per_text():
    generator, calls = stub_generator(lambda prefix: b"x ")
    assert generator.generate(["a ", "b "], max_generated_words=2, seed=10) == ["x x ", "x x "]
    assert [seeds for _, seeds in calls] == [[10, 11], [12, 13]]  # Distinct across texts and steps


def test_generation_refuses_prompts_beyond_the_context():
    generator, _ = stub_generator(lambda prefix: b"x ", max_model_len=4)
    with pytest.raises(ContextTooLongError):
        generator.generate(["a b c"], max_generated_words=2)
