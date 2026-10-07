"""
Train WeLT with Megatron-Bridge.

    torchrun --nproc_per_node=<gpus> -m welt_training.train config.yaml [section.key=value ...]

The YAML has a `model` and a `data` section (see `welt_training/experiments`), and optionally any of the
Megatron-Bridge config sections (`train`, `optimizer`, `scheduler`, `validation`, `checkpoint`, `logger`, `ddp`,
`rng`), whose keys are set as-is on the corresponding Megatron-Bridge config.
"""
import math
import os
import sys
from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.core.rerun_state_machine import get_rerun_state_machine
from utf8_tokenizer.tokenizer import UTF8Tokenizer

from welt.model import WeLTModelProvider
from welt_training.data import WeLTDatasetProvider
from welt_training.extendable_yaml import CONFIG_FILE_NAME, load_yaml

TOKENIZER = UTF8Tokenizer()


def build_dataset_provider(model: dict, data: dict) -> WeLTDatasetProvider:
    return WeLTDatasetProvider(render_images=model.get("image_encoder") is not None,
                               pretokenizer_name=model.get("pretokenizer"),
                               trust_remote_code=model.get("trust_remote_code", False),
                               **data)


def build_config(config: dict, model_provider, dataset_provider, vocab_size: int) -> ConfigContainer:
    """Megatron-Bridge config from the YAML's Megatron-Bridge sections, with the given model and dataset."""
    output_dir = config.get("output_dir", "./output")

    def section(cls, name: str, **defaults):
        return cls(**defaults | (config.get(name) or {}))  # Unknown keys raise a TypeError

    train = section(TrainingConfig, "train", train_iters=1000, micro_batch_size=32, global_batch_size=32)
    optimizer = section(OptimizerConfig, "optimizer", optimizer="adam", lr=3e-4, min_lr=3e-5, weight_decay=0.01,
                        bf16=True, adam_beta1=0.9, adam_beta2=0.95, clip_grad=1.0, use_distributed_optimizer=True)
    scheduler = section(SchedulerConfig, "scheduler", lr_decay_style="cosine", lr_warmup_iters=0,
                        lr_decay_iters=train.train_iters, start_weight_decay=optimizer.weight_decay,
                        end_weight_decay=optimizer.weight_decay, weight_decay_incr_style="constant")
    validation = section(ValidationConfig, "validation", eval_interval=500, eval_iters=10)
    checkpoint = section(CheckpointConfig, "checkpoint", save=os.path.join(output_dir, "checkpoints"),
                         load=os.path.join(output_dir, "checkpoints"), save_interval=1000, ckpt_format="torch_dist")
    logger = section(LoggerConfig, "logger", log_interval=10, tensorboard_dir=os.path.join(output_dir, "tensorboard"))
    ddp = section(DistributedDataParallelConfig, "ddp", use_distributed_optimizer=optimizer.use_distributed_optimizer,
                  grad_reduce_in_fp32=True, average_in_collective=False, overlap_grad_reduce=True,
                  overlap_param_gather=True)
    rng = section(RNGConfig, "rng", seed=42)

    model_provider.seq_length = dataset_provider.seq_length
    model_provider.calculate_per_token_loss = True
    model_provider.bf16 = optimizer.bf16
    dataset_provider.samples_per_eval = validation.eval_iters * train.global_batch_size

    return ConfigContainer(
        model=model_provider,
        dataset=dataset_provider,
        train=train,
        optimizer=optimizer,
        scheduler=scheduler,
        validation=validation,
        checkpoint=checkpoint,
        logger=logger,
        ddp=ddp,
        rng=rng,
        # Datasets tokenize on their own, Megatron only needs the vocabulary size
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=vocab_size),
    )


def report(value: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """A logged metric: Megatron sums both over micro batches and data parallel ranks, and logs their ratio."""
    return torch.stack([value.detach().float(), count.float()])


def to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def loss_func(losses: torch.Tensor, correct: torch.Tensor, labels: torch.Tensor):
    """Per-byte cross entropy, plus bits per byte and byte/word accuracy for logging.
    labels: (words, bytes). Bits per byte count every prediction (including the EOS ending each word) except the
    EOS ending a document (a word with an empty label), per UTF-8 byte of text, like the causal LM baseline."""
    loss_mask = labels != TOKENIZER.pad_token_id
    eos = labels == TOKENIZER.eos_token_id
    document_end = eos & F.pad(torch.ones_like(eos[:, :1]), (0, eos.size(1) - 1), value=False)
    losses = losses.float()

    loss = (losses * loss_mask).sum()
    num_tokens = loss_mask.sum().int()
    get_rerun_state_machine().validate_result(result=loss, rejection_func=torch.isnan,
                                              message="found NaN in local forward loss calculation",
                                              tolerance=0.0, fatal=True)

    words = loss_mask.any(dim=-1)
    words_correct = (correct | ~loss_mask).all(dim=-1) & words
    return loss, num_tokens, {
        "lm loss": report(loss, num_tokens),
        "bits per byte": report((losses * (loss_mask & ~document_end)).sum() / math.log(2), (loss_mask & ~eos).sum()),
        "byte accuracy": report((correct & loss_mask).sum(), num_tokens),
        "word accuracy": report(words_correct.sum(), words.sum()),
    }


def forward_step(state, data_iterator, model, return_schedule_plan: bool = False):
    batch = to_cuda(next(data_iterator))
    losses, correct, labels = model(**batch)
    return losses, partial(loss_func, correct=correct, labels=labels)


def run(config: dict, cfg: ConfigContainer, forward_step_func, save_artifacts=None):
    """Save the YAML config (and other artifacts) to the output directory, then train."""
    output_dir = config.get("output_dir", "./output")
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, CONFIG_FILE_NAME), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        if save_artifacts is not None:
            save_artifacts(output_dir)

    pretrain(config=cfg, forward_step_func=forward_step_func)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def main(build, forward_step_func):
    """Train from a YAML config and overrides on the command line, with build(config) -> (ConfigContainer, artifacts
    saving function)."""
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <config.yaml> [section.key=value ...]")
    config = load_yaml(sys.argv[1], sys.argv[2:])
    cfg, save_artifacts = build(config)
    run(config, cfg, forward_step_func, save_artifacts)


def build(config: dict):
    # Model options other than the transformers (and the pretokenizer, a data option) are provider fields
    model = WeLTModelProvider.from_hf(**{key: value for key, value in config["model"].items() if key != "pretokenizer"})
    dataset = build_dataset_provider(config["model"], config["data"])
    cfg = build_config(config, model, dataset, vocab_size=model.num_tokens)
    return cfg, lambda output_dir: dataset.processor().save_pretrained(os.path.join(output_dir, "processor"))


if __name__ == "__main__":
    main(build, forward_step)
