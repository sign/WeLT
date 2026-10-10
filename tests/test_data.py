from itertools import islice

import pytest
from datasets import Dataset

from welt_training.data_utils import PAD_WORD, TextDataConfig, build_iterators, pack_words, train_texts


def test_pack_words_fills_blocks_and_pads():
    packed = list(pack_words([["a", "b"], ["c", "d", "e"], ["f", "g", "h", "i"]], seq_length=5))
    assert packed == [(["a", "b", "c", "d", "e"], [2, 3]), (["f", "g", "h", "i", PAD_WORD], [4, 1])]


def test_pack_words_splits_long_sequences():
    packed = list(pack_words([list("abcdefg")], seq_length=4))
    assert packed == [(list("abcd"), [4]), ([*"efg", PAD_WORD], [3, 1])]


def json_config(tmp_path, columns: dict, **kwargs) -> TextDataConfig:
    path = tmp_path / "data.jsonl"
    Dataset.from_dict(columns).to_json(path)
    return TextDataConfig(seq_length=16, dataset_name="json", data_files=str(path), **kwargs)


def test_validation_is_held_out_from_train(tmp_path):
    config = json_config(tmp_path, {"text": [f"text {i}" for i in range(10)]}, max_eval_samples=3)
    assert [e["text"] for e in config.texts("validation")] == ["text 0", "text 1", "text 2"]
    assert sorted(e["text"] for e in config.texts("train")) == sorted(f"text {i}" for i in range(3, 10))


def test_texts_from_a_template(tmp_path):
    config = json_config(tmp_path, {"src": ["s0", "s1"], "tgt": ["t0", "t1"]},
                         dataset_text_template=["<{src}> ", "{tgt}"], max_eval_samples=1)
    assert [e["text"] for e in config.texts("validation")] == ["<s0> t0"]


def test_texts_require_a_text_column(tmp_path):
    config = json_config(tmp_path, {"src": ["a", "b"]}, max_eval_samples=1)
    with pytest.raises(ValueError, match="No 'text' column"):
        list(config.texts("validation"))


def test_train_texts_are_endless_and_split_across_ranks(tmp_path):
    config = json_config(tmp_path, {"text": [f"t{i}" for i in range(10)]}, max_eval_samples=2)
    ranks = [list(islice(train_texts(config, rank, world_size=2), 8)) for rank in range(2)]
    assert sorted(ranks[0][:4] + ranks[1][:4]) == sorted(f"t{i}" for i in range(2, 10))  # One epoch, split
    assert sorted(ranks[0][4:] + ranks[1][4:]) == sorted(ranks[0][:4] + ranks[1][:4])  # The next epoch


def test_build_iterators_batches_examples(tmp_path):
    config = json_config(tmp_path, {"text": [f"t{i}" for i in range(10)]}, max_eval_samples=3)
    config.num_workers, config.pin_memory, config.persistent_workers = 2, False, False
    config.micro_batch_size, config.eval_micro_batches = 2, 3

    def make_examples(texts):
        yield from ({"text": text} for text in texts)

    train, validation = build_iterators(config, make_examples, list, rank=0, world_size=1)
    batches = list(islice(train, 4))
    assert all(len(batch) == 2 for batch in batches)
    assert {e["text"] for batch in batches for e in batch} <= {f"t{i}" for i in range(3, 10)}
    # Each evaluation sees the validation examples (repeated to fill its micro batches), the same ones every time
    evaluations = [[next(validation) for _ in range(3)] for _ in range(2)]
    assert evaluations[0] == evaluations[1]
    assert evaluations[0][:2] == [[{"text": "t0"}, {"text": "t1"}], [{"text": "t2"}]]
