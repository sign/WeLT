import pytest
from datasets import Dataset

from welt_training.data_utils import PAD_WORD, TextDataConfig, load_text_datasets, pack_words


def test_pack_words_fills_blocks_and_pads():
    packed = pack_words({"words": [["a", "b"], ["c", "d", "e"], ["f", "g", "h", "i"]]}, seq_length=5)
    assert packed["words"] == [["a", "b", "c", "d", "e"], ["f", "g", "h", "i", PAD_WORD]]
    assert packed["seq_lengths"] == [[2, 3], [4, 1]]


def test_pack_words_truncates_long_sequences():
    packed = pack_words({"words": [list("abcdefg")]}, seq_length=4)
    assert packed["words"] == [list("abcd")]
    assert packed["seq_lengths"] == [[4]]


def test_load_text_datasets_with_template(tmp_path):
    path = tmp_path / "data.json"
    Dataset.from_dict({"src": [f"s{i}" for i in range(20)], "tgt": [f"t{i}" for i in range(20)]}).to_json(path)
    args = TextDataConfig(seq_length=16, train_file=str(path), dataset_text_template=["<{src}> ", "{tgt}"],
                               validation_split_percentage=10)
    texts = load_text_datasets(args)
    assert len(texts["train"]) == 18
    assert len(texts["validation"]) == 2
    assert texts["train"].column_names == ["text"]
    assert all(t.startswith("<s") and " t" in t for t in texts["train"]["text"])


def test_dataset_lengths_repeat_examples_equally():
    from types import SimpleNamespace

    from welt_training.data_utils import dataset_lengths

    datasets = {"train": [0] * 10, "validation": [0] * 4}
    context = SimpleNamespace(train_samples=15, valid_samples=100)
    assert dataset_lengths(datasets, context, samples_per_eval=None) == {"train": 20, "validation": 100}
    # Each evaluation (one epoch of the sampler) covers the validation set
    assert dataset_lengths(datasets, context, samples_per_eval=8) == {"train": 20, "validation": 8}


def test_words_dataset_collates_examples():
    pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
    from welt_training.data import WeLTDatasetProvider, WordsDataset

    processor = WeLTDatasetProvider(seq_length=8).processor()
    dataset = Dataset.from_dict({"words": [["\x02", "a"]], "seq_lengths": [[2]]})
    words = WordsDataset(dataset, processor, length=5)
    assert len(words) == 5
    batch = words.collate_fn([words[i] for i in range(3)])
    assert batch["input_ids"].shape[0] == 3
