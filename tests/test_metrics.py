"""Tests for compute_bits_per_byte metric utility."""

import math

import pytest

from welt_training.metrics import compute_bits_per_byte


class TestComputeBitsPerByte:
    """Tests for the compute_bits_per_byte function."""

    def test_basic_computation(self):
        """BPB = loss * num_tokens / (num_bytes * ln(2))."""
        loss = 2.0
        num_tokens = 100
        num_bytes = 100
        expected = 2.0 / math.log(2)
        assert compute_bits_per_byte(loss, num_tokens, num_bytes) == pytest.approx(expected)

    def test_byte_level_model(self):
        """When tokens == bytes (byte-level model), BPB = loss / ln(2)."""
        loss = 1.5
        bpb = compute_bits_per_byte(loss, num_tokens=1000, num_bytes=1000)
        assert bpb == pytest.approx(loss / math.log(2))

    def test_subword_tokenizer(self):
        """When tokens < bytes (subword tokenizer), BPB accounts for compression."""
        loss = 3.0
        num_tokens = 50
        num_bytes = 200  # 4 bytes per token on average
        expected = 3.0 * 50 / (200 * math.log(2))
        assert compute_bits_per_byte(loss, num_tokens, num_bytes) == pytest.approx(expected)

    def test_zero_loss(self):
        """Zero loss should give zero BPB (perfect prediction)."""
        assert compute_bits_per_byte(0.0, 100, 100) == 0.0

    def test_zero_bytes_returns_inf(self):
        """Zero bytes should return infinity."""
        result = compute_bits_per_byte(1.0, 100, 0)
        assert result == float("inf")

    def test_zero_tokens_returns_zero(self):
        """Zero tokens should return zero BPB."""
        result = compute_bits_per_byte(1.0, 0, 100)
        assert result == 0.0

    def test_higher_loss_means_higher_bpb(self):
        """Higher loss should give higher BPB."""
        bpb_low = compute_bits_per_byte(1.0, 100, 100)
        bpb_high = compute_bits_per_byte(2.0, 100, 100)
        assert bpb_high > bpb_low

    def test_more_bytes_means_lower_bpb(self):
        """More bytes for same total bits should give lower BPB."""
        bpb_fewer = compute_bits_per_byte(1.0, 100, 100)
        bpb_more = compute_bits_per_byte(1.0, 100, 200)
        assert bpb_more < bpb_fewer

    def test_known_value_one_bit_per_byte(self):
        """loss = ln(2) nats per token, 1 token, 1 byte -> 1.0 bit per byte."""
        loss = math.log(2)
        assert compute_bits_per_byte(loss, 1, 1) == pytest.approx(1.0)

    def test_relationship_with_perplexity(self):
        """BPB = log2(perplexity) when num_tokens == num_bytes."""
        loss = 2.5
        perplexity = math.exp(loss)
        bpb = compute_bits_per_byte(loss, num_tokens=1, num_bytes=1)
        assert bpb == pytest.approx(math.log2(perplexity))

    def test_scaling_with_token_count(self):
        """Doubling num_tokens with same num_bytes should double BPB."""
        bpb1 = compute_bits_per_byte(1.0, num_tokens=100, num_bytes=200)
        bpb2 = compute_bits_per_byte(1.0, num_tokens=200, num_bytes=200)
        assert bpb2 == pytest.approx(2 * bpb1)


class TestBPBTokenizerIntegration:
    """Validate BPB computation patterns used in the training scripts."""

    def test_byte_level_model_pattern(self):
        """
        Validate the train.py (WeLTTrainer) pattern.

        WeLT is a byte-level model: each token IS a byte, so num_tokens == num_bytes.
        The trainer passes num_tokens=1, num_bytes=1 to get the ratio BPB = loss / ln(2).
        Verify this matches log2(perplexity).
        """
        loss = 4.2
        perplexity = math.exp(loss)

        # This is exactly what WeLTTrainer._add_custom_metrics does
        bpb = compute_bits_per_byte(loss, num_tokens=1, num_bytes=1)

        assert bpb == pytest.approx(loss / math.log(2))
        assert bpb == pytest.approx(math.log2(perplexity))

    def test_subword_tokenizer_compression_ratio(self):
        """
        Validate the run_clm.py pattern where tokens < bytes due to subword tokenization.

        For a BPE tokenizer, each token covers ~3-4 bytes on average.
        BPB must account for this compression:
            BPB = loss_per_token * (num_tokens / num_bytes) / ln(2)
        """
        loss = 3.0

        # Byte-level baseline: 1 token per byte
        bpb_byte = compute_bits_per_byte(loss, num_tokens=400, num_bytes=400)

        # BPE tokenizer: ~3.5 bytes per token (typical for English)
        bpb_bpe = compute_bits_per_byte(loss, num_tokens=400, num_bytes=1400)

        # BPE should produce lower BPB (same loss spread over more bytes)
        assert bpb_bpe < bpb_byte
        assert bpb_bpe == pytest.approx(bpb_byte * 400 / 1400)

    def test_decode_roundtrip_byte_counting(self):
        """
        Validate the run_clm.py approach: decode token IDs → encode as UTF-8 → count bytes.

        Uses a real HuggingFace tokenizer to verify the decode-based byte counting
        correctly captures the tokenizer's compression ratio.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine translation is an important NLP task.",
        ]

        # Simulate run_clm.py pipeline: tokenize, then decode to count bytes
        all_token_ids = []
        for text in texts:
            all_token_ids.extend(tokenizer.encode(text))

        num_tokens = len(all_token_ids)
        original_bytes = sum(len(t.encode("utf-8")) for t in texts)

        # run_clm.py decodes per-chunk; simulate with full sequence
        decoded_text = tokenizer.decode(all_token_ids)
        decoded_bytes = len(decoded_text.encode("utf-8"))

        # Decoded bytes should closely match original (tokenizer roundtrip)
        assert abs(decoded_bytes - original_bytes) <= len(texts), \
            f"Decoded bytes ({decoded_bytes}) should match original ({original_bytes})"

        # Subword tokenizer: fewer tokens than bytes
        assert num_tokens < original_bytes, \
            "BPE tokenizer should compress text (fewer tokens than bytes)"

        # Verify BPB with a known loss
        loss = 2.5
        bpb = compute_bits_per_byte(loss, num_tokens, decoded_bytes)

        # Must be less than byte-level BPB (since tokens < bytes)
        bpb_byte_level = loss / math.log(2)
        assert bpb < bpb_byte_level

        # Must equal the analytical formula
        expected = loss * num_tokens / (decoded_bytes * math.log(2))
        assert bpb == pytest.approx(expected)
