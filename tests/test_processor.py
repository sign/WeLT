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
    inputs = processor(text, packed=True)
    assert torch.equal(inputs["input_ids"][0], torch.tensor([[2, 2, 3, 0], [2, 97, 32, 3], [2, 98, 32, 3]]))
    assert inputs["input_attention_mask"][0].shape == (3, 4)
    assert inputs["attention_mask"][0].shape == (1, 3, 3)
    assert torch.equal(inputs["position_ids"][0], torch.tensor([0, 1, 2]))
    assert torch.equal(inputs["labels_input"][0], torch.tensor([[2, 97, 32, 98], [2, 98, 3, 0], [2, 3, 0, 0]]))
    assert torch.equal(inputs["labels_output"][0], torch.tensor([[97, 32, 98, 3], [98, 3, 0, 0], [3, 0, 0, 0]]))


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


def test_processor_packed_vs_unpacked_labels(processor):
    text = "hello world test"

    # Test packed=True (default)
    inputs_packed = processor(text, collated=True, packed=True)

    # Test packed=False
    inputs_unpacked = processor(text, collated=True, packed=False)

    # Both should have same structure
    assert all(key in inputs_packed for key in expected_keys)
    assert all(key in inputs_unpacked for key in expected_keys)

    # Input tokens should be the same
    assert torch.equal(inputs_packed["input_ids"], inputs_unpacked["input_ids"])
    assert torch.equal(inputs_packed["attention_mask"], inputs_unpacked["attention_mask"])

    # Labels should be different due to different packing strategies
    assert not torch.equal(inputs_packed["labels_input"], inputs_unpacked["labels_input"])
    assert not torch.equal(inputs_packed["labels_output"], inputs_unpacked["labels_output"])


def test_processor_packed_false_default_behavior(processor):
    text = "example text for testing"

    # Default should be packed=True
    inputs_default = processor(text, collated=True)
    inputs_explicit_packed = processor(text, collated=True, packed=False)

    # Should be identical
    assert torch.equal(inputs_default["labels_input"], inputs_explicit_packed["labels_input"])
    assert torch.equal(inputs_default["labels_output"], inputs_explicit_packed["labels_output"])


def test_get_words_and_labels_packed_vs_unpacked(processor):
    text = "hello world test"
    words = processor.pretokenize(text)

    # Test packed=True
    labels_packed = processor.get_sequence_labels(words, pack=True)

    # Test packed=False
    labels_unpacked = processor.get_sequence_labels(words, pack=False)

    # Labels should be different
    assert labels_packed != labels_unpacked

    assert labels_packed == ['hello world test', 'world test', 'test', '']
    assert labels_unpacked == ['hello ', 'world ', 'test', '']


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
        " "  # Space is added by pretokenize
    ]


def test_pretokenize_multiple_whitespace(processor):
    text = """
    def foo():
        return "bar"
    """.strip()
    words = processor.pretokenize(text)
    assert words == [ControlTokens.StartOfText, "def ", "foo():\n", " " * 8, 'return ', '"bar" ']


def test_get_words_and_labels_packed_vs_unpacked_respect_max_word_length(processor, renderer):
    text = "this is a long-test"
    words = processor.pretokenize(text)

    new_processor = TextImageProcessor(
        pretokenizer=WordsSegmentationTokenizer(),
        tokenizer=processor.tokenizer,
        renderer=renderer,
        image_processor=processor.image_processor,
        max_word_length=3
    )

    # Test packed=True
    labels_packed = new_processor.get_sequence_labels(words, pack=True)

    # Test packed=False
    labels_unpacked = new_processor.get_sequence_labels(words, pack=False)

    assert labels_packed == ['thi', 'is', 'a l', 'lon', '']
    assert labels_unpacked == ['thi', 'is ', 'a ', 'lon', '']


def test_pretokenize_dataset(processor):
    texts = [
        "hi!",
        "hello world",
    ]
    dataset = Dataset.from_dict({"text": texts})
    dataset = processor.pretokenize_dataset(dataset)

    assert dataset[:] == {
        'words': [
            [ControlTokens.StartOfText, 'hi! '],
            [ControlTokens.StartOfText, 'hello ', 'world '],
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
                ControlTokens.StartOfText, 'a ', 'b ', 'c ',
                ControlTokens.StartOfText, 'hello ', 'world ',
            ],
            [
                ControlTokens.StartOfText, 'hi! ',
                ControlTokens.StartOfText, 'yes. ',
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
    labels = processor.get_sequence_labels(datum["words"], datum["seq_lengths"], pack=True)

    assert labels == [
        'a b', 'b', '',
        'c d', 'd', ''
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


def test_processor_save_and_load_works_without_image_processor():
    processor = TextImageProcessor(
        pretokenizer=WordsSegmentationTokenizer(),
        tokenizer=UTF8Tokenizer(),
        renderer=renderer,
        image_processor=NoopImageProcessor())

    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        print(temp_dir)
        processor.save_pretrained(save_directory=temp_dir, push_to_hub=False)
        new_processor = TextImageProcessor.from_pretrained(temp_dir)
        assert isinstance(new_processor.image_processor, NoopImageProcessor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
