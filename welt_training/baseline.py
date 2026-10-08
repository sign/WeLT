"""
Causal language model baseline: the same data as WeLT, through a standard tokenizer (subword, or bytes) and a single
transformer, trained with Megatron-Bridge. Reports bits per byte, comparable to WeLT's.

    torchrun --nproc_per_node=<gpus> -m welt_training.baseline config.yaml [section.key=value ...]

The YAML is like WeLT's (see `experiments/machine-translation/baseline.yaml`), with a `model` section of:
    transformer: a HF model id/path, or a JSON HF config
    tokenizer: a HF tokenizer id/path
    load_pretrained: whether to initialize the transformer from its HF weights
and `data.seq_length` counting tokens. Documents are concatenated (each after an EOS) and split into chunks.
Like WeLT, it does not predict (or score) the text of shift blocks (`\x0E...\x0F`, e.g. the source sentence).
"""
import math
import re
from dataclasses import dataclass
from functools import partial

import torch
from datasets import Dataset
from megatron.bridge import AutoBridge
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from transformers import AutoTokenizer

from welt.model import safetensors_checkpoint, transformer_provider
from welt_training.data_utils import TextDataConfig, dataset_lengths, load_text_datasets
from welt_training.train import build_config, main, report, to_cuda


def token_byte_lengths(tokenizer) -> torch.Tensor:
    """UTF-8 bytes of the text of each token id, 0 for special tokens. Exact for byte-level BPE vocabularies
    (GPT-2, Pythia, Llama 3, Qwen) and SentencePiece ones ("▁" for spaces, <0xNN> byte fallback; their dummy
    prefix space counts as a byte)."""
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    byte_level = tokenizer.is_fast and '"ByteLevel"' in tokenizer.backend_tokenizer.to_str()
    special = set(tokenizer.all_special_ids)
    lengths = []
    for token_id, token in enumerate(vocab):
        if token_id in special or token is None:
            lengths.append(0)
        elif byte_level:
            lengths.append(len(token))  # ByteLevel maps each byte to one character
        elif re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token):
            lengths.append(1)
        else:
            lengths.append(len(token.replace("▁", " ").encode("utf-8")))
    return torch.tensor(lengths, dtype=torch.long)


SHIFT_BLOCK = re.compile("\x0e[^\x0f]*\x0f")


def chunk_tokens(batch: dict[str, list], tokenizer, length: int) -> dict[str, list]:
    """Tokenize documents, concatenate them (each after an EOS), and split into chunks of `length` tokens.
    Chunks overlap by one token, so that every token is a label exactly once; the last one is padded with EOS.
    loss_mask is 0 for the padding, and for the tokens within shift blocks (after \x0E, up to \x0F)."""
    encoded = tokenizer(batch["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids, mask = [], []
    for text, document, offsets in zip(batch["text"], encoded.input_ids, encoded.offset_mapping, strict=True):
        blocks = [(match.start(), match.end()) for match in SHIFT_BLOCK.finditer(text)]
        ids += [tokenizer.eos_token_id, *document]
        mask += [1, *(int(not any(start + 1 < end <= block_end for start, block_end in blocks))
                      for _, end in offsets)]
    padding = -(len(ids) - 1) % (length - 1)
    ids, mask = ids + [tokenizer.eos_token_id] * padding, mask + [0] * padding
    starts = range(0, len(ids) - 1, length - 1)
    return {"input_ids": [ids[i:i + length] for i in starts], "loss_mask": [mask[i:i + length] for i in starts]}


class TokensDataset(torch.utils.data.Dataset):
    """Token chunks as inputs, labels, their loss mask, and the UTF-8 bytes of each (scored) label.
    Repeats the examples up to length."""

    def __init__(self, dataset: Dataset, byte_lengths: torch.Tensor, length: int):
        self.dataset = dataset
        self.byte_lengths = byte_lengths
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        example = self.dataset[int(index) % len(self.dataset)]
        ids, loss_mask = torch.tensor(example["input_ids"]), torch.tensor(example["loss_mask"][1:])
        return {"input_ids": ids[:-1], "labels": ids[1:], "loss_mask": loss_mask,
                "label_bytes": self.byte_lengths[ids[1:]] * loss_mask}


@dataclass(kw_only=True)
class TokensDatasetProvider(TextDataConfig, DatasetProvider):
    tokenizer_name: str

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


def loss_func(losses: torch.Tensor, loss_mask: torch.Tensor, label_bytes: torch.Tensor):
    """Per-token cross entropy, plus bits per byte over the scored text tokens (not EOS)."""
    losses = losses.float()
    loss = (losses * loss_mask).sum()
    num_tokens = loss_mask.sum().int()
    return loss, num_tokens, {
        "lm loss": report(loss, num_tokens),
        "bits per byte": report((losses * (label_bytes > 0)).sum() / math.log(2), label_bytes.sum()),
    }


def forward_step(state, data_iterator, model, return_schedule_plan: bool = False):
    batch = to_cuda(next(data_iterator))
    input_ids = batch["input_ids"]
    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)
    losses = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=batch["labels"])
    return losses, partial(loss_func, loss_mask=batch["loss_mask"], label_bytes=batch["label_bytes"])


def build(config: dict):
    """The Megatron-Bridge config of a baseline YAML config."""
    model_config = config["model"]
    trust_remote_code = model_config.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer"], trust_remote_code=trust_remote_code)

    transformer = model_config["transformer"]
    if model_config.get("load_pretrained", False):
        model = AutoBridge.from_hf_pretrained(safetensors_checkpoint(transformer), trust_remote_code=trust_remote_code
                                              ).to_megatron_provider(load_weights=True)
    else:
        model = transformer_provider(transformer, trust_remote_code)
    model.vocab_size = max(model.vocab_size, len(tokenizer))  # HF embeddings may have more (padding) rows

    dataset = TokensDatasetProvider(tokenizer_name=model_config["tokenizer"], trust_remote_code=trust_remote_code,
                                    **config["data"])
    return build_config(config, model, dataset, vocab_size=model.vocab_size), None


if __name__ == "__main__":
    main(build, forward_step)
