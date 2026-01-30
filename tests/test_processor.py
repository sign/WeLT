import pickle
import tempfile

import pytest
import torch
from datasets import Dataset
from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS
from pixel_renderer import PixelRendererProcessor
from trl.data_utils import pack_dataset
from utf8_tokenizer.control import ControlTokens
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from words_segmentation.tokenizer import WordsSegmentationTokenizer

from tests.test_model import setup_tiny_model
from welt.noop import NoopImageProcessor
from welt.processor import TextImageProcessor


@pytest.fixture(scope="module")
def processor():
    model, processor, collator = setup_tiny_model()
    return processor


@pytest.fixture(scope="module")
def renderer():
    font_config = FontConfig(sources=FONTS_NOTO_SANS)
    return PixelRendererProcessor(font=font_config)


expected_tensor_keys = ["input_ids", "input_attention_mask", "attention_mask", "position_ids",
                        "labels_input", "labels_attention_mask", "labels_output",
                        "input_images", "input_images_dimensions"]
expected_keys = expected_tensor_keys


def test_processor_multiprocessing_pickle(processor):
    # Processor should be pickleable for multiprocessing
    pickle.dumps(processor)


def test_processor_single_text_collated(processor):
    text = "example text for testing"
    inputs = processor(text, collated=True)

    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], torch.Tensor) for key in expected_tensor_keys)


def test_processor_single_text_not_collated(processor):
    text = "example text for testing"
    inputs = processor(text)
    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], list) and len(inputs[key]) == 1 for key in expected_tensor_keys)


def test_processor_single_text_value(processor):
    text = "a b"
    inputs = processor(text)
    assert torch.equal(inputs["input_ids"][0], torch.tensor([[2, 2, 3, 0], [2, 97, 32, 3], [2, 98, 3, 0]]))
    assert inputs["input_attention_mask"][0].shape == (3, 4)
    assert inputs["attention_mask"][0].shape == (1, 3, 3)
    assert torch.equal(inputs["position_ids"][0], torch.tensor([0, 1, 2]))
    # Unpacked mode: labels are shorter (only next token, not all remaining)
    assert torch.equal(inputs["labels_input"][0], torch.tensor([[2, 97, 32], [2, 98, 3], [2, 3, 0]]))
    assert torch.equal(inputs["labels_output"][0], torch.tensor([[97, 32, 3], [98, 3, 0], [3, 0, 0]]))


def test_processor_list_format_collated(processor):
    text = "example text for testing"
    inputs = processor([text], collated=True)
    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], torch.Tensor) for key in expected_tensor_keys)


def test_processor_object_format_collated(processor):
    text = "example text for testing"
    inputs = processor({"text": text}, collated=True)
    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], torch.Tensor) for key in expected_tensor_keys)


def test_processor_multiple_strings_collated_attention_mask(processor):
    texts = ["one", "two words", "three word test"]
    inputs = processor(texts, collated=True)
    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], torch.Tensor) for key in expected_tensor_keys)

    assert inputs["attention_mask"].shape == (3, 1, 4, 4)

    expected = [
        torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False]
        ]),
        torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [False, False, False, False]
        ]),
        torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True]
        ])
    ]

    for mask, expected_mask in zip(inputs["attention_mask"], expected, strict=False):
        assert torch.equal(mask[0], expected_mask)


def test_get_words_and_labels(processor):
    text = "hello world test"
    words = processor.pretokenize(text)

    labels = processor.get_sequence_labels(words)

    # Unpacked mode: each token predicts only the next token
    assert labels == ['hello ', 'world ', 'test', '']


def test_render_images_shape(processor):
    texts = ["short", "a bit longer text"]
    renders, dimensions = processor.render_texts(texts)

    assert renders.shape == (2, 3, 16, 112)
    assert torch.equal(dimensions, torch.tensor([[16, 48], [16, 112]]))


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


def test_get_words_and_labels_respect_max_word_length(processor, renderer):
    text = "this is a long-test"

    new_processor = TextImageProcessor(
        pretokenizer=WordsSegmentationTokenizer(max_bytes=3),
        tokenizer=processor.tokenizer,
        renderer=renderer,
        image_processor=processor.image_processor,
    )

    words = new_processor.pretokenize(text)
    labels = new_processor.get_sequence_labels(words)

    # max_bytes=3 truncates words during pretokenization
    assert words == [ControlTokens.StartOfText, 'thi', 's ', 'is ', 'a ', 'lon', 'g-t', 'est']
    # Unpacked mode: each token predicts the next token
    assert labels == ['thi', 's ', 'is ', 'a ', 'lon', 'g-t', 'est', '']


def test_pretokenize_dataset(processor):
    texts = [
        "hi!",
        "hello world",
    ]
    dataset = Dataset.from_dict({"text": texts})
    dataset = processor.pretokenize_dataset(dataset)

    assert dataset[:] == {
        'words': [
            [ControlTokens.StartOfText, 'hi!'],
            [ControlTokens.StartOfText, 'hello ', 'world'],
        ],
    }


def test_packed_dataset(processor):
    texts = [
        "hi!",
        "hello world",
        "yes.",
        "a b c"
    ]
    dataset = Dataset.from_dict({"text": texts})
    dataset = processor.pretokenize_dataset(dataset)
    packed_dataset = pack_dataset(dataset, seq_length=7)

    assert packed_dataset[:] == {
        'seq_lengths': [
            [4, 3],
            [2, 2],
        ],
        'words': [
            [
                ControlTokens.StartOfText, 'a ', 'b ', 'c',
                ControlTokens.StartOfText, 'hello ', 'world',
            ],
            [
                ControlTokens.StartOfText, 'hi!',
                ControlTokens.StartOfText, 'yes.',
            ],
        ],
    }


def test_packed_dataset_labels_independent(processor):
    texts = [
        "a b",
        "c d",
    ]
    dataset = Dataset.from_dict({"text": texts})
    dataset = processor.pretokenize_dataset(dataset)
    packed_dataset = pack_dataset(dataset, seq_length=8)

    datum = next(iter(packed_dataset))
    labels = processor.get_sequence_labels(datum["words"], datum["seq_lengths"])

    # Unpacked mode: each token predicts only the next token, respecting sequence boundaries
    assert labels == [
        'a ', 'b', '',
        'c ', 'd', ''
    ]


def test_processor_works_on_packed_sequence(processor):
    texts = [
        "hi!",
        "hello world",
        "yes.",
        "a b c"
    ]
    dataset = Dataset.from_dict({"text": texts})
    dataset = processor.pretokenize_dataset(dataset)
    packed_dataset = pack_dataset(dataset, seq_length=8)

    transformed_dataset = packed_dataset.with_transform(processor)
    for inputs in transformed_dataset:
        assert all(key in inputs for key in expected_keys)
        assert all(isinstance(inputs[key], torch.Tensor) for key in expected_tensor_keys)


def test_processor_save_and_load_works(processor):
    with tempfile.TemporaryDirectory() as temp_dir:
        processor.save_pretrained(save_directory=temp_dir, push_to_hub=False)
        new_processor = TextImageProcessor.from_pretrained(temp_dir)

        for attr in processor.attributes:
            assert getattr(new_processor, attr) is not None
            assert getattr(new_processor, attr).__class__.__name__ == getattr(processor, attr).__class__.__name__


def test_processor_save_and_load_works_without_image_processor(renderer):
    processor = TextImageProcessor(
        pretokenizer=WordsSegmentationTokenizer(),
        tokenizer=UTF8Tokenizer(),
        renderer=renderer,
        image_processor=NoopImageProcessor())

    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        processor.save_pretrained(save_directory=temp_dir, push_to_hub=False)
        new_processor = TextImageProcessor.from_pretrained(temp_dir)
        assert isinstance(new_processor.image_processor, NoopImageProcessor)


def test_labels_masked_in_shift_blocks(renderer):
    """Test that labels are zeroed for tokens inside shift blocks (except ShiftIn itself)."""
    processor = TextImageProcessor(
        pretokenizer=WordsSegmentationTokenizer(),
        tokenizer=UTF8Tokenizer(),
        renderer=renderer,
        image_processor=NoopImageProcessor())

    # Use f-string template and let processor segment into words
    text = f"<en>{ControlTokens.ShiftOut}hello{ControlTokens.ShiftIn}<he> שלום"
    words = processor.pretokenize(text)

    # Expected words: BOS, "<en>", SO, "hello", SI, "<he> ", "שלום"
    # Labels inside shift blocks (SO and "hello") should be zeroed in process_single_example
    # ShiftIn keeps its label to predict next word

    result = processor.process_single_example(words, [len(words)])

    # Masked positions (inside shift block): indices 2, 3 (SO and "hello")
    # These should have zero labels (PAD tokens)
    for idx in [2, 3]:
        assert result["labels_input"][idx].sum() == 0, f"labels_input at {idx} should be all zeros"
        assert result["labels_attention_mask"][idx].sum() == 0, f"labels_attention_mask at {idx} should be all zeros"

    # Non-masked positions should have non-zero labels
    for idx in [0, 1, 4, 5]:
        assert result["labels_input"][idx].sum() != 0
        assert result["labels_attention_mask"][idx].sum() != 0


def test_multiple_shift_blocks(renderer):
    """Test handling of multiple shift blocks in a sequence."""
    processor = TextImageProcessor(
        pretokenizer=WordsSegmentationTokenizer(),
        tokenizer=UTF8Tokenizer(),
        renderer=renderer,
        image_processor=NoopImageProcessor())

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
