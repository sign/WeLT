from itertools import islice

import pytest
import torch
from datasets import Dataset
from utf8_tokenizer.tokenizer import EOS_TOKEN_ID, PAD_TOKEN, PAD_TOKEN_ID

from welt_training import data_utils
from welt_training.data_utils import TextDataConfig, build_iterators, pack_words, train_texts


def test_pack_words_fills_blocks_and_pads():
    packed = list(pack_words([["a", "b"], ["c", "d", "e"], ["f", "g", "h", "i"]], seq_length=5))
    assert packed == [(["a", "b", "c", "d", "e"], [2, 3], [False, False]),
                      (["f", "g", "h", "i", PAD_TOKEN], [4, 1], [False, False])]


def test_pack_words_splits_long_sequences():
    # Chunks overlap by one word: "d" ends the first (without a label) and starts the second
    packed = list(pack_words([list("abcdefg")], seq_length=4))
    assert packed == [(list("abcd"), [4], [True]), (list("defg"), [4], [False])]
    packed = list(pack_words([list("abcde"), ["f"]], seq_length=4))
    assert packed == [(list("abcd"), [4], [True]), ([*"de", "f", PAD_TOKEN], [2, 1, 1], [False, False, False])]


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


def test_texts_from_token_ids(tmp_path):
    from transformers import AutoTokenizer

    tokens = AutoTokenizer.from_pretrained("gpt2")("Hello world, שלום").input_ids
    config = json_config(tmp_path, {"tokens": [tokens, tokens]}, dataset_tokenizer="gpt2", max_eval_samples=1)
    assert [e["text"] for e in config.texts("validation")] == ["Hello world, שלום"]


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

    def collate(examples):
        return {"texts": [example["text"] for example in examples]}

    train, validation = build_iterators(config, make_examples, collate, rank=0, world_size=1)
    batches = list(islice(train, 4))
    assert all(len(batch["texts"]) == 2 for batch in batches)
    assert {text for batch in batches for text in batch["texts"]} <= {f"t{i}" for i in range(3, 10)}
    # Each evaluation sees every validation example once, filled up with copies that count for nothing (score 0),
    # the same ones every time
    evaluations = [[next(validation) for _ in range(3)] for _ in range(2)]
    assert [[(batch["texts"], batch["score"].item()) for batch in evaluation] for evaluation in evaluations] == [
        [(["t0", "t1"], 1), (["t2"], 1), (["t0", "t1"], 0)]] * 2


def test_build_iterators_refuse_to_evaluate_a_subset(tmp_path):
    config = json_config(tmp_path, {"text": [f"t{i}" for i in range(10)]}, max_eval_samples=5)
    config.num_workers, config.pin_memory, config.persistent_workers = 0, False, False
    config.micro_batch_size, config.eval_micro_batches = 2, 2
    with pytest.raises(ValueError, match="increase validation.eval_iters"):
        build_iterators(config, lambda texts: ({"text": text} for text in texts),
                        lambda examples: {"n": len(examples)}, rank=0, world_size=1)


def test_texts_drop_nul_bytes_and_empty_texts(tmp_path):
    config = json_config(tmp_path, {"text": ["a\x00b", "", "\x00", "c"]}, max_eval_samples=4)
    assert [e["text"] for e in config.texts("validation")] == ["ab", "c"]
    config = json_config(tmp_path, {"text": ["a\x00b", "", "\x00", "c"]}, max_eval_samples=0)
    assert sorted(islice(train_texts(config, rank=0, world_size=1), 2)) == ["ab", "c"]


def test_train_texts_are_split_across_dataloader_workers(tmp_path, monkeypatch):
    """Each dataloader worker keeps every num_workers-th text of its rank: together, one epoch."""
    config = json_config(tmp_path, {"text": [f"t{i}" for i in range(10)]}, max_eval_samples=0)
    workers = []
    for worker_id in range(2):
        info = type("WorkerInfo", (), {"id": worker_id, "num_workers": 2})
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda info=info: info)
        workers.append(list(islice(train_texts(config, rank=0, world_size=1), 5)))
    assert not set(workers[0]) & set(workers[1])
    assert sorted(workers[0] + workers[1]) == sorted(f"t{i}" for i in range(10))


def test_baseline_counts_the_bytes_of_its_texts():
    """Each token counts the text since the previous one: every byte once, also with characters split across tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    texts = ["Hello wörld 👋 שלום 日本語 🧑‍🤝‍🧑", "a"]
    (ids, _, nbytes), = data_utils.token_chunks(texts, tokenizer, length=64)
    assert sum(nbytes) == sum(len(text.encode()) for text in texts)
    assert [n for i, n in zip(ids, nbytes, strict=True) if i == tokenizer.eos_token_id] == [0] * ids.count(
        tokenizer.eos_token_id)


@pytest.mark.parametrize("seq_length", [4, 5, 6, 8])
def test_baseline_scores_the_bytes_welt_scores(seq_length):
    """Every token is a label once (chunks overlap by one token); shift blocks are not scored, as in WeLT."""
    from transformers import AutoTokenizer

    from welt.processor import TextImageProcessor, next_word_labels

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    texts = ["<en>\x0eHello world\x0f<he> שלום עולם", "no shift block here", "<a>\x0eb\x0f c"]
    examples = list(data_utils.token_examples(texts, tokenizer, length=4))
    labels = [token for example in examples for token in example["labels"][example["loss_mask"].bool()].tolist()]
    # The labels: each document (after the first, after its EOS) without the text of its shift block
    assert tokenizer.decode(labels) == tokenizer.eos_token.join(data_utils.SHIFT_BLOCK.sub("\x0e", t) for t in texts)
    baseline_bytes = sum(int(example["label_bytes"].sum()) for example in examples)

    processor = TextImageProcessor.create(max_word_length=32, render_images=False)
    welt_bytes = 0
    # Packed in short examples: documents are split across examples, also within shift blocks, as long documents are
    # in training
    for packed in pack_words(map(processor.pretokenize, texts), seq_length=seq_length):
        example = processor.process_single_example(*packed)
        inputs = (example[k][None] for k in ("input_ids", "sequence_ids", "label_mask"))
        labels = next_word_labels(*inputs)
        welt_bytes += int(((labels != PAD_TOKEN_ID) & (labels != EOS_TOKEN_ID)).sum())
    assert baseline_bytes == welt_bytes == sum(len(data_utils.SHIFT_BLOCK.sub("\x0e", t).encode()) for t in texts)
