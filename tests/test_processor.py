import pickle
import tempfile

import pytest
import torch
from utf8_tokenizer.control import ControlTokens
from words_segmentation.tokenizer import WordsSegmentationTokenizer

from tests.conftest import ORACLE_TEXTS, PAD, SI, SO, oracle_labels
from welt.processor import TextImageProcessor, collate_fn, get_shift_blocks, next_word_labels
from welt_training.data_utils import pack_words


@pytest.fixture(scope="module")
def processor():
    return TextImageProcessor.create(max_word_length=32, render_images=True)


@pytest.fixture(scope="module")
def text_processor():
    return TextImageProcessor.create(max_word_length=32, render_images=False)


expected_tensor_keys = ["input_ids", "sequence_ids", "block_ids", "label_mask", "input_patches", "input_patches_shape"]


def labels_of(example: dict) -> torch.Tensor:
    """Each word's label bytes (bytes, EOS), PAD for words without a label."""
    return next_word_labels(example["input_ids"][None], example["sequence_ids"][None], example["label_mask"][None])[0]


def test_processor_multiprocessing_pickle(processor):
    # Processor should be pickleable for multiprocessing
    pickle.dumps(processor)


def test_packed_padding_does_not_add_eos_labels(processor):
    words = processor.pretokenize("a b")
    original = processor.process_single_example(words, [len(words)])
    padded = processor.process_single_example(
        words + [PAD] * 5, [len(words)] + [1] * 5)
    assert torch.equal(labels_of(padded)[:len(words)], labels_of(original))
    assert not padded["label_mask"][len(words):].any()
    assert (labels_of(padded)[len(words):] == processor.tokenizer.pad_token_id).all()


def test_processor_single_text_value(processor):
    inputs = processor(["a b"])
    assert torch.equal(inputs["input_ids"][0], torch.tensor([[2, 2, 3, 0], [2, 97, 32, 3], [2, 98, 3, 0]]))
    assert inputs["sequence_ids"][0].tolist() == [1, 1, 1]
    assert inputs["block_ids"][0].tolist() == [0, 0, 0]
    # Each word predicts the next one; the last one, the end of the text (an empty word)
    assert torch.equal(labels_of({k: v[0] for k, v in inputs.items()}),
                       torch.tensor([[97, 32, 3], [98, 3, 0], [3, 0, 0]]))


def test_patch_positions():
    from welt.patches import patch_positions

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

    # max_bytes=3 truncates words during pretokenization
    assert words == [ControlTokens.StartOfText, 'thi', 's ', 'is ', 'a ', 'lon', 'g-t', 'est']


def test_packed_dataset_labels_independent(processor):
    texts = [
        "a b",
        "c d",
    ]
    words, seq_lengths, continues = next(pack_words(map(processor.pretokenize, texts), seq_length=8))
    example = processor.process_single_example(words, seq_lengths, continues)
    labels = [bytes(label[label > 3].tolist()).decode() if mask else None
              for label, mask in zip(labels_of(example), example["label_mask"], strict=True)]

    # Each word predicts only the next word of its sequence (the last one, an empty word);
    # packing pads with PAD words, each an isolated sequence without a label
    assert labels == [
        'a ', 'b', '',
        'c ', 'd', '',
        None, None,
    ]
    assert example["sequence_ids"].tolist() == [1, 1, 1, 2, 2, 2, 3, 4]


def test_processor_save_and_load_works(processor):
    with tempfile.TemporaryDirectory() as temp_dir:
        processor.save_pretrained(temp_dir)
        new_processor = TextImageProcessor.from_pretrained(temp_dir)
        assert new_processor.renderer is not None
        assert new_processor.max_word_length == processor.max_word_length
        assert new_processor.pretokenize("hello world") == processor.pretokenize("hello world")
        # Words longer than max_word_length are split alike (words-segmentation does not save max_bytes)
        long_text = "internationalization " * 3
        assert new_processor.pretokenize(long_text) == processor.pretokenize(long_text)


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

    # ShiftOut and the words inside blocks have no label (the blocks attend bidirectionally, ShiftIn included)
    assert result["label_mask"].int().tolist() == [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    assert result["block_ids"].tolist() == [0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0]
    labels = labels_of(result)
    assert (labels[~result["label_mask"]] == 0).all()


def test_shift_blocks_do_not_span_packed_sequences(processor):
    """A block cut between two sequences is two: to the end of the first, and from the start of the second."""
    words = ["a", SO, "b", SI, "c"]
    result = processor.process_single_example(words, [3, 2])
    assert result["block_ids"].tolist() == [0, 1, 1, 2, 0]
    assert result["label_mask"].tolist() == [True, False, True, True, True]


def test_bpe_pretokenizer_words_are_text_spans():
    """Byte-level BPE tokens ("Ġworld") are encoded, words are the spans of text they cover."""
    processor = TextImageProcessor.create(max_word_length=32, render_images=False,
                                          pretokenizer_name="EleutherAI/pythia-14m")
    text = f"<text>{ControlTokens.ShiftOut}héllo world{ControlTokens.ShiftIn} שלום"
    words = processor.pretokenize(text)
    assert "".join(words) == ControlTokens.StartOfText + text
    assert ControlTokens.ShiftOut in words
    assert ControlTokens.ShiftIn in words


def test_bpe_pretokenizer_splits_long_tokens():
    """Tokens longer than max_word_length (with BOS and EOS), e.g. Pythia's runs of spaces, are split, not truncated."""
    processor = TextImageProcessor.create(max_word_length=8, render_images=False,
                                          pretokenizer_name="EleutherAI/pythia-14m")
    text = "a" + " " * 20 + "שלוםשלום"
    words = processor.pretokenize(text)
    assert "".join(words) == ControlTokens.StartOfText + text
    assert all(len(word.encode()) <= 6 for word in words)


def test_collate_fn_pads_every_dimension_and_keeps_dtypes():
    batch = [{"mask": torch.tensor([[True, False]]), "ids": torch.tensor([1])},
             {"mask": torch.tensor([[True], [True]]), "ids": torch.tensor([2, 3])}]
    collated = collate_fn(batch)
    assert collated["mask"].dtype == torch.bool
    assert torch.equal(collated["mask"], torch.tensor([[[True, False], [False, False]],
                                                       [[True, False], [True, False]]]))
    assert torch.equal(collated["ids"], torch.tensor([[1, 0], [2, 3]]))


@pytest.mark.parametrize("seq_length", [5, 8, 16, 40, 128])  # Short: documents split across examples
def test_labels_match_an_oracle_of_the_words(text_processor, seq_length):
    """The labels the model derives (next_word_labels) on packed examples, against labels from the words alone."""
    documents = [text_processor.pretokenize(text) for text in ORACLE_TEXTS * 2]
    all_labels = []
    for words, seq_lengths, continues in pack_words(documents, seq_length):
        example = text_processor.process_single_example(words, seq_lengths, continues)
        expected = oracle_labels(words, seq_lengths, continues)
        all_labels += [label for label in expected if label is not None]
        assert example["label_mask"].tolist() == [label is not None for label in expected]
        if not example["label_mask"].any():  # A chunk within a shift block
            continue
        labels = labels_of(example)[example["label_mask"]]
        tokenized = text_processor.tokenize_words([label for label in expected if label is not None]).input_ids[:, 1:]
        assert torch.equal(labels[:, :tokenized.size(1)], tokenized)
        assert (labels[:, tokenized.size(1):] == 0).all()
    # Documents split across examples: every word's label appears exactly once, as in the unsplit documents
    assert all_labels == [label for document in documents for label in oracle_labels(document, [len(document)])
                          if label is not None]


@pytest.mark.parametrize(("words", "blocks"), [
    (["a", "b", "c"], []),
    (["a", SO, "b", SI, "c"], [(1, 3)]),
    ([SO, SI, "c"], [(0, 1)]),
    (["a", SO, "b", SI, "c", SO, "d", SI], [(1, 3), (5, 7)]),
    (["a", SO, "b"], [(1, 2)]),  # Cut at the end of a sequence: to its last word
    (["a", SI, "b"], [(0, 1)]),  # Cut at its start: from its first word
    (["a", SO, "b", SO, "c", SI], [(1, 5)]),  # A ShiftOut within a block belongs to it (as the baseline's regex)
])
def test_shift_blocks(words, blocks):
    """Shift blocks span ShiftOut to ShiftIn, inclusive."""
    assert list(get_shift_blocks(words)) == blocks
