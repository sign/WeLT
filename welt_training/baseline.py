"""
Causal language model baseline: the same data as WeLT, through a standard tokenizer (subword, or bytes) and a single
transformer, trained with Megatron-Bridge. Reports bits per byte, comparable to WeLT's.

    torchrun --nproc_per_node=<gpus> -m welt_training.baseline config.yaml [section.key=value ...]

The YAML is like WeLT's (see `experiments/machine-translation/baseline.yaml`), with a `model` section of:
    transformer: a HF model id/path, or a JSON HF config
    tokenizer: a HF tokenizer id/path
    load_pretrained: whether to initialize the transformer from its HF weights
and `data.seq_length` counting tokens. Documents are concatenated (separated by EOS) and split into chunks.
"""
import math
from dataclasses import dataclass
from functools import partial

import torch
from datasets import Dataset
from megatron.bridge import AutoBridge
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from transformers import AutoTokenizer

from welt.model import hf_config, safetensors_checkpoint
from welt_training.data_utils import TextDataConfig, dataset_lengths, load_text_datasets
from welt_training.train import build_config, main, report, to_cuda


def token_byte_lengths(tokenizer) -> torch.Tensor:
    """UTF-8 bytes of the text of each token id, 0 for special tokens."""
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    byte_level = decoder is not None and type(decoder).__name__ == "ByteLevel"
    special = set(tokenizer.all_special_ids)
    lengths = []
    for token_id, token in enumerate(vocab):
        if token_id in special or token is None:
            lengths.append(0)
        elif byte_level:
            lengths.append(len(token))  # ByteLevel maps each byte to one character
        else:
            lengths.append(len(tokenizer.convert_tokens_to_string([token]).encode("utf-8")))
    return torch.tensor(lengths, dtype=torch.long)


def chunk_tokens(batch: dict[str, list], tokenizer, length: int) -> dict[str, list]:
    """Tokenize documents, concatenate them separated by EOS, and split into chunks of `length` tokens."""
    ids = [token for ids in tokenizer(batch["text"], add_special_tokens=False).input_ids
           for token in [*ids, tokenizer.eos_token_id]]
    return {"input_ids": [ids[i:i + length] for i in range(0, len(ids) - length + 1, length)]}


class TokensDataset(torch.utils.data.Dataset):
    """Token chunks as inputs, labels, and the UTF-8 bytes of each label. Repeats the examples up to length."""

    def __init__(self, dataset: Dataset, byte_lengths: torch.Tensor, length: int):
        self.dataset = dataset
        self.byte_lengths = byte_lengths
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ids = torch.tensor(self.dataset[int(index) % len(self.dataset)]["input_ids"])
        return {"input_ids": ids[:-1], "labels": ids[1:], "label_bytes": self.byte_lengths[ids[1:]]}


@dataclass(kw_only=True)
class TokensDatasetProvider(TextDataConfig, DatasetProvider):
    tokenizer_name: str
    dataloader_type: str = "cyclic"

    def build_datasets(self, context: DatasetBuildContext):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=self.trust_remote_code)
        byte_lengths = token_byte_lengths(tokenizer)
        texts = load_text_datasets(self)
        datasets = {split: texts[split].map(chunk_tokens, batched=True, remove_columns=["text"],
                                            fn_kwargs={"tokenizer": tokenizer, "length": self.seq_length + 1},
                                            num_proc=self.preprocessing_num_workers, desc=f"Tokenizing {split}")
                    for split in texts}
        lengths = dataset_lengths(datasets, context, self.samples_per_eval)
        return (*(TokensDataset(datasets[split], byte_lengths, lengths[split]) if split in datasets else None
                  for split in ("train", "validation")), None)


def loss_func(losses: torch.Tensor, label_bytes: torch.Tensor):
    """Per-token cross entropy, plus bits per byte over the content tokens (not EOS)."""
    losses = losses.float()
    loss = losses.sum()
    num_tokens = torch.tensor(losses.numel(), device=losses.device, dtype=torch.int)
    content = label_bytes > 0

    return loss, num_tokens, {
        "lm loss": report(loss, num_tokens),
        "bits per byte": report((losses * content).sum() / math.log(2), label_bytes.sum()),
    }


def forward_step(state, data_iterator, model, return_schedule_plan: bool = False):
    batch = to_cuda(next(data_iterator))
    losses = model(input_ids=batch["input_ids"], position_ids=None, attention_mask=None, labels=batch["labels"])
    return losses, partial(loss_func, label_bytes=batch["label_bytes"])


def build(config: dict):
    """The Megatron-Bridge config of a baseline YAML config."""
    model_config = config["model"]
    trust_remote_code = model_config.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer"], trust_remote_code=trust_remote_code)

    transformer = model_config["transformer"]
    load_pretrained = model_config.get("load_pretrained", False)
    if load_pretrained:
        bridge = AutoBridge.from_hf_pretrained(safetensors_checkpoint(transformer), trust_remote_code=trust_remote_code)
    else:
        bridge = AutoBridge.from_hf_config(hf_config(transformer, trust_remote_code))
    model = bridge.to_megatron_provider(load_weights=load_pretrained)
    model.vocab_size = len(tokenizer)

    dataset = TokensDatasetProvider(tokenizer_name=model_config["tokenizer"], trust_remote_code=trust_remote_code,
                                    **config["data"])
    return build_config(config, model, dataset, vocab_size=len(tokenizer)), None


if __name__ == "__main__":
    main(build, forward_step)
