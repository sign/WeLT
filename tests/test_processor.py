import pickle
import tempfile

import pytest
import torch
from datasets import Dataset
from utf8_tokenizer.control import ControlTokens
from words_segmentation.tokenizer import WordsSegmentationTokenizer

from welt.processor import TextImageProcessor
from welt_training.data_utils import pack_dataset


@pytest.fixture(scope="module")
def processor():
    return TextImageProcessor.create(max_word_length=32, render_images=True)


@pytest.fixture(scope="module")
def text_processor():
    return TextImageProcessor.create(max_word_length=32, render_images=False)


expected_tensor_keys = ["input_ids", "input_attention_mask", "attention_mask",
                        "labels_input", "labels_attention_mask", "labels_output",
                        "input_patches", "input_patches_shape"]


def test_processor_multiprocessing_pickle(processor):
    # Processor should be pickleable for multiprocessing
    pickle.dumps(processor)


def test_packed_padding_does_not_add_eos_labels(processor):
    words = processor.pretokenize("a b")
    original = processor.process_single_example(words, [len(words)])
    padded = processor.process_single_example(
        words + [processor.tokenizer.pad_token] * 5, [len(words)] + [1] * 5)
    assert torch.equal(padded["labels_output"][:len(words)], original["labels_output"])
    assert (padded["labels_output"][len(words):] == processor.tokenizer.pad_token_id).all()
    assert (padded["labels_attention_mask"][len(words):] == 0).all()


def test_processor_single_text_value(processor):
    inputs = processor(["a b"])
    assert torch.equal(inputs["input_ids"][0], torch.tensor([[2, 2, 3, 0], [2, 97, 32, 3], [2, 98, 3, 0]]))
    assert inputs["input_attention_mask"][0].shape == (3, 4)
    assert inputs["attention_mask"][0].shape == (1, 3, 3)
    # Unpacked mode: labels are shorter (only next token, not all remaining)
    assert torch.equal(inputs["labels_input"][0], torch.tensor([[2, 97, 32], [2, 98, 3], [2, 3, 0]]))
    assert torch.equal(inputs["labels_output"][0], torch.tensor([[97, 32, 3], [98, 3, 0], [3, 0, 0]]))


def test_patch_positions():
    from welt.processor import patch_positions

    rows, columns = patch_positions(torch.tensor([[1, 3], [2, 2]]))  # A 1x3 image, then a 2x2 one
    assert rows.tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert columns.tolist() == [0, 1, 2, 0, 1, 0, 1]


def test_render_images_shape(processor):
    texts = ["short", "a bit longer text"]
    patches, shapes = processor.render_texts(texts)
    # 16px high renders, split into 16x16 RGB patches, packed: 3 then 7
    assert patches.shape == (3 + 7, 16 * 16 * 3)
    assert patches.dtype == torch.uint8
    assert torch.equal(shapes, torch.tensor([[1, 3], [1, 7]]))  # (rows, columns) of patches


def test_pretokenize_splits_control_tokens(processor):
    text = (f"{ControlTokens.ShiftOut}test{ControlTokens.ShiftIn}"
            f"{ControlTokens.StartOfHeading}hello {ControlTokens.EndOfText}")
    words = processor.pretokenize(text)
    assert words == [
        ControlTokens.StartOfText,  # BOS is added by pretokenize
        ControlTokens.ShiftOut, 'test', ControlTokens.ShiftIn,
        ControlTokens.StartOfHeading, "hello ", ControlTokens.EndOfText,
    ]


def test_pretokenize_multiple_whitespace(processor):
    text = """
    def foo():
        return "bar"
    """.strip()
    words = processor.pretokenize(text)
    assert words == [ControlTokens.StartOfText, "def ", "foo():\n", " " * 8, 'return ', '"bar"']


def test_get_words_and_labels_respect_max_word_length(processor):
    text = "this is a long-test"

    new_processor = TextImageProcessor(
        pretokenizer=WordsSegmentationTokenizer(max_bytes=3), renderer=None)

    words = new_processor.pretokenize(text)
    labels = new_processor.get_sequence_labels(words, [len(words)])

    # max_bytes=3 truncates words during pretokenization
    assert words == [ControlTokens.StartOfText, 'thi', 's ', 'is ', 'a ', 'lon', 'g-t', 'est']
    # Unpacked mode: each token predicts the next token
    assert labels == ['thi', 's ', 'is ', 'a ', 'lon', 'g-t', 'est', '']


def test_packed_dataset(processor):
    texts = [
        "hi!",
        "hello world",
        "yes.",
        "a b c"
    ]
    dataset = Dataset.from_dict({"text": texts})
    packed_dataset = pack_dataset(processor, dataset, seq_length=7)

    pad = "\x00"
    assert packed_dataset[:] == {
        'seq_lengths': [
            [2, 3, 2],
            [4, 1, 1, 1],
        ],
        'words': [
            [
                ControlTokens.StartOfText, 'hi!',
                ControlTokens.StartOfText, 'hello ', 'world',
                ControlTokens.StartOfText, 'yes.',
            ],
            [
                ControlTokens.StartOfText, 'a ', 'b ', 'c', pad, pad, pad,
            ],
        ],
    }


def test_packed_dataset_labels_independent(processor):
    texts = [
        "a b",
        "c d",
    ]
    dataset = Dataset.from_dict({"text": texts})
    packed_dataset = pack_dataset(processor, dataset, seq_length=8)

    datum = next(iter(packed_dataset))
    labels = processor.get_sequence_labels(datum["words"], datum["seq_lengths"])

    # Unpacked mode: each token predicts only the next token, respecting sequence boundaries
    # Packing pads with PAD words, each an isolated sequence with an empty label
    assert labels == [
        'a ', 'b', '',
        'c ', 'd', '',
        '', '',
    ]


def test_processor_save_and_load_works(processor):
    with tempfile.TemporaryDirectory() as temp_dir:
        processor.save_pretrained(temp_dir)
        new_processor = TextImageProcessor.from_pretrained(temp_dir)
        assert new_processor.renderer is not None
        assert new_processor.max_word_length == processor.max_word_length
        assert new_processor.pretokenize("hello world") == processor.pretokenize("hello world")


def test_processor_save_and_load_works_without_renderer(text_processor):
    with tempfile.TemporaryDirectory() as temp_dir:
        text_processor.save_pretrained(temp_dir)
        new_processor = TextImageProcessor.from_pretrained(temp_dir)
        assert new_processor.renderer is None
        assert "input_patches" not in new_processor(["hello"])


def test_multiple_shift_blocks():
    """Test handling of multiple shift blocks in a sequence."""
    processor = TextImageProcessor(pretokenizer=WordsSegmentationTokenizer(), renderer=None)

    words = [
        ControlTokens.StartOfText,
        ControlTokens.ShiftOut, "first", "block", ControlTokens.ShiftIn,
        "middle", "token",
        ControlTokens.ShiftOut, "second", "block", ControlTokens.ShiftIn,
        "end"
    ]

    result = processor.process_single_example(words, [len(words)])

    # ShiftOut and content inside blocks should have zeroed labels
    # First block: indices 1, 2, 3 (ShiftOut, "first", "block")
    # Second block: indices 7, 8, 9 (ShiftOut, "second", "block")
    masked_indices = [1, 2, 3, 7, 8, 9]
    for idx in masked_indices:
        assert result["labels_input"][idx].sum() == 0, f"labels_input at {idx} should be all zeros"
        assert result["labels_attention_mask"][idx].sum() == 0, f"labels_attention_mask at {idx} should be all zeros"

    # Non-masked positions should have non-zero labels (except last position which has empty label)
    non_masked_indices = [0, 4, 5, 6, 10]
    for idx in non_masked_indices:
        assert result["labels_input"][idx].sum() != 0
        assert result["labels_attention_mask"][idx].sum() != 0


def test_bpe_pretokenizer_words_are_text_spans():
    """Byte-level BPE tokens ("Ġworld") are encoded, words are the spans of text they cover."""
    processor = TextImageProcessor.create(max_word_length=32, render_images=False,
                                          pretokenizer_name="EleutherAI/pythia-14m")
    text = f"<text>{ControlTokens.ShiftOut}héllo world{ControlTokens.ShiftIn} שלום"
    words = processor.pretokenize(text)
    assert "".join(words) == ControlTokens.StartOfText + text
    assert ControlTokens.ShiftOut in words
    assert ControlTokens.ShiftIn in words
