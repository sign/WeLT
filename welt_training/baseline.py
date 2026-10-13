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
from dataclasses import dataclass
from functools import partial

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.core import parallel_state
from transformers import AutoTokenizer

from welt.model import safetensors_checkpoint, transformer_provider
from welt_training.data_utils import TextDataConfig, build_iterators, token_examples
from welt_training.train import build_config, main, report, to_cuda


@dataclass(kw_only=True)
class TokensDatasetProvider(TextDataConfig, DatasetProvider):
    tokenizer_name: str
    dataloader_type: str = "external"  # Iterators of micro batches (build_iterators)

    def build_datasets(self, context: DatasetBuildContext):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=self.trust_remote_code)
        make_examples = partial(token_examples, tokenizer=tokenizer, length=self.seq_length)
        train, validation = build_iterators(self, make_examples, torch.utils.data.default_collate,
                                            rank=parallel_state.get_data_parallel_rank(),
                                            world_size=parallel_state.get_data_parallel_world_size())
        return train, validation, None


def loss_func(losses: torch.Tensor, loss_mask: torch.Tensor, label_bytes: torch.Tensor, text_mask: torch.Tensor,
              score: torch.Tensor | None = None):
    """Per-token cross entropy, plus bits per byte over the scored text tokens (not EOS). score: as WeLT's."""
    if score is not None:
        loss_mask, label_bytes, text_mask = loss_mask * score, label_bytes * score, text_mask * score
    losses = losses.float()
    loss, num_tokens, metrics = masked_next_token_loss(loss_mask, losses)  # With NaN / Inf checks
    return loss, num_tokens, metrics | {
        "bits per byte": report((losses * text_mask).sum() / math.log(2), label_bytes.sum()),
    }


def forward_step(state, data_iterator, model, return_schedule_plan: bool = False):
    batch = to_cuda(next(data_iterator))
    score = batch.pop("score", None)
    input_ids = batch["input_ids"]
    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)
    losses = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=batch["labels"])
    return losses, partial(loss_func, loss_mask=batch["loss_mask"], label_bytes=batch["label_bytes"],
                           text_mask=batch["text_mask"], score=score)


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
    # HF embeddings may have more (padding) rows. Padded to a multiple of 128 (as Pythia's 50,304) for tensor cores, and
    # with Transformer Engine's fused cross entropy: together 2x faster steps with a vocabulary-sized LM head
    model.vocab_size = -(-max(model.vocab_size, len(tokenizer)) // 128) * 128
    model.cross_entropy_fusion_impl = "te"
    # Other model options are the transformer's (e.g. hidden_dropout), as WeLT's are its latent transformer's
    for key, value in model_config.items():
        if key not in ("transformer", "tokenizer", "load_pretrained", "trust_remote_code"):
            if not hasattr(model, key):
                raise ValueError(f"Unknown model option {key!r}")
            setattr(model, key, value)

    dataset = TokensDatasetProvider(tokenizer_name=model_config["tokenizer"],
                                    **{"trust_remote_code": trust_remote_code} | config["data"])
    return build_config(config, model, dataset, vocab_size=model.vocab_size), None


if __name__ == "__main__":
    main(build, forward_step)
