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


def build_model_provider(model: dict, data: dict) -> WeLTModelProvider:
    model = dict(model)
    provider = WeLTModelProvider.from_hf(
        bytes_encoder=model.pop("bytes_encoder", None),
        image_encoder=model.pop("image_encoder", None),
        latent_transformer=model.pop("latent_transformer"),
        bytes_decoder=model.pop("bytes_decoder"),
        load_pretrained=model.pop("load_pretrained", False),
        trust_remote_code=model.pop("trust_remote_code", False),
    )
    provider.seq_length = data["seq_length"]
    provider.calculate_per_token_loss = True
    for key, value in model.items():  # e.g. modality_dropout, recompute_granularity
        setattr(provider, key, value)
    return provider


def build_dataset_provider(model: dict, data: dict) -> WeLTDatasetProvider:
    data = dict(data)
    return WeLTDatasetProvider(render_images=model.get("image_encoder") is not None,
                               pretokenizer_name=model.get("pretokenizer"),
                               trust_remote_code=model.get("trust_remote_code", False),
                               **data)


def _apply(config_object, values: dict | None):
    for key, value in (values or {}).items():
        if not hasattr(config_object, key):
            raise ValueError(f"Unknown {type(config_object).__name__} option: {key}")
        setattr(config_object, key, value)
    return config_object


def build_config(config: dict, model_provider, dataset_provider, vocab_size: int) -> ConfigContainer:
    """Megatron-Bridge config from the YAML's Megatron-Bridge sections, with the given model and dataset."""
    output_dir = config.get("output_dir", "./output")
    model_provider.bf16 = True

    train = _apply(TrainingConfig(train_iters=1000, micro_batch_size=32, global_batch_size=32), config.get("train"))
    optimizer = _apply(OptimizerConfig(optimizer="adam", lr=3e-4, min_lr=3e-5, weight_decay=0.01, bf16=True,
                                       adam_beta1=0.9, adam_beta2=0.95, clip_grad=1.0,
                                       use_distributed_optimizer=True), config.get("optimizer"))
    scheduler = _apply(SchedulerConfig(lr_decay_style="cosine", lr_warmup_iters=0, lr_decay_iters=train.train_iters,
                                       start_weight_decay=optimizer.weight_decay,
                                       end_weight_decay=optimizer.weight_decay,
                                       weight_decay_incr_style="constant"), config.get("scheduler"))
    validation = _apply(ValidationConfig(eval_interval=500, eval_iters=10), config.get("validation"))
    checkpoint = _apply(CheckpointConfig(save=os.path.join(output_dir, "checkpoints"),
                                         load=os.path.join(output_dir, "checkpoints"),
                                         save_interval=1000, ckpt_format="torch_dist"), config.get("checkpoint"))
    logger = _apply(LoggerConfig(log_interval=10, tensorboard_dir=os.path.join(output_dir, "tensorboard")),
                    config.get("logger"))
    ddp = _apply(DistributedDataParallelConfig(use_distributed_optimizer=optimizer.use_distributed_optimizer,
                                               grad_reduce_in_fp32=True, average_in_collective=False,
                                               overlap_grad_reduce=True, overlap_param_gather=True),
                 config.get("ddp"))
    rng = _apply(RNGConfig(seed=42), config.get("rng"))

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


def loss_func(losses: torch.Tensor, correct: torch.Tensor, labels: torch.Tensor):
    """Per-byte cross entropy, plus bits-per-byte (without EOS) and byte/word accuracy for logging."""
    loss_mask = labels != TOKENIZER.pad_token_id
    content_mask = loss_mask & (labels != TOKENIZER.eos_token_id)
    losses = losses.float()

    loss = (losses * loss_mask).sum()
    num_tokens = loss_mask.sum().int()
    get_rerun_state_machine().validate_result(result=loss, rejection_func=torch.isnan,
                                              message="found NaN in local forward loss calculation",
                                              tolerance=0.0, fatal=True)

    def report(value, count):
        return torch.stack([value.detach().float(), count.float()])

    words = loss_mask.any(dim=-1)
    words_correct = (correct | ~loss_mask).all(dim=-1) & words
    return loss, num_tokens, {
        "lm loss": report(loss, num_tokens),
        "bits per byte": report((losses * content_mask).sum() / math.log(2), content_mask.sum()),
        "byte accuracy": report((correct & loss_mask).sum(), num_tokens),
        "word accuracy": report(words_correct.sum(), words.sum()),
    }


def forward_step(state, data_iterator, model, return_schedule_plan: bool = False):
    batch = {key: value.cuda(non_blocking=True) for key, value in next(data_iterator).items()}
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


def train(args: list[str] | None = None):
    args = sys.argv[1:] if args is None else args
    config = load_yaml(args[0], args[1:])
    model = build_model_provider(config["model"], config["data"])
    dataset = build_dataset_provider(config["model"], config["data"])
    cfg = build_config(config, model, dataset, vocab_size=model.num_tokens)
    run(config, cfg, forward_step,
        save_artifacts=lambda output_dir: dataset.processor().save_pretrained(os.path.join(output_dir, "processor")))


if __name__ == "__main__":
    train()
