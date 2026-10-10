# Training

```shell
torchrun --nproc_per_node=<gpus> -m welt_training.train <config.yaml> [section.key=value ...]
```

Overrides are YAML values, at any depth: `output_dir=./output/test`, `train.train_iters=50`,
`logger.wandb_project=null` (to run without W&B), `model.hidden_dropout=0.1`.
Configs can inherit with `$extends: ./other.yaml` (relative to the file), deep-merging their sections.

## Config

A config has a `model` and a `data` section, plus any Megatron-Bridge section. The keys, as in
[`string-repetition.yaml`](experiments/easy-tasks/string-repetition.yaml):

```yaml
output_dir: ...           # The run directory
model:
  bytes_encoder: ...      # HF model id/path, a JSON HF config, or null
  image_encoder: ...      # HF model id/path, a JSON HF config, or null
  latent_transformer: ...
  bytes_decoder: ...
  load_pretrained: ...    # Initialize id/path transformers from their HF weights
  pretokenizer: ...       # A HF tokenizer splitting words, defaults to sign/words-segmentation
  trust_remote_code: ...
  hidden_dropout: ...     # Any other WeLTModelProvider field, e.g. recompute_granularity
data:
  dataset_name: ...
  dataset_config_name: ...
  dataset_text_template:  # [prefix, completion], or one string
  seq_length: ...         # Words per packed example
  max_word_length: ...    # Bytes per word, including BOS and EOS
# Megatron-Bridge sections, keys set as-is on its configs (unknown keys raise)
train:                    # TrainingConfig
optimizer:                # OptimizerConfig
scheduler:                # SchedulerConfig
validation:               # ValidationConfig
checkpoint:               # CheckpointConfig
logger:                   # LoggerConfig, wandb_project: null disables W&B, wandb_exp_name names the run
ddp:                      # DistributedDataParallelConfig
rng:                      # RNGConfig
```

Defaults ([`train.py`](train.py) `build_config`): 1000 iterations of batch 32, distributed Adam
(lr 3e-4 → 3e-5 cosine, betas 0.9/0.95, weight decay 0.01, grad clip 1.0), bf16, evaluation every 500 iterations,
checkpoints every 1000, seed 42.

The run directory (`output_dir`) holds the config (`welt.yaml`), the `processor/`, `tensorboard/` logs and Megatron
`checkpoints/`, from which a rerun resumes automatically.

### Model

`model` keys other than the four transformers, `load_pretrained`, `pretokenizer` and `trust_remote_code` are set on
the `WeLTModelProvider` ([`welt/model.py`](../welt/model.py)), the latent transformer's Megatron config, e.g.
`hidden_dropout` or `recompute_granularity`.
Precision and recomputation settings are shared with the other transformers
(`SHARED_CONFIG_FIELDS`); other fields, e.g. `hidden_dropout`, only apply to the latent transformer.

For the image encoder (described in the [README](../README.md#model-setup)), see
[`ocr.yaml`](experiments/easy-tasks/ocr.yaml) and [`image-encoder-tiny.json`](experiments/models/image-encoder-tiny.json).

### Data

The `data` section configures the `WeLTDatasetProvider` ([`data.py`](data.py), [`data_utils.py`](data_utils.py)).
Texts are streamed from a HF dataset (`dataset_name`, `dataset_config_name`), or from local files with
`dataset_name: json` (or `text`, `csv`, ...) and `data_files` (a path, or `{train: ..., validation: ...}`).
Validation is the first `max_eval_samples` (default 256) examples of the validation split, or, without one, held out
from the start of the train split.

`dataset_text_template` is a Python format string over the dataset's columns; without it, the `text` column is used
(a missing `text` column raises). As `[prefix, completion]`, both parts are concatenated for training, and
`welt-evaluate` generates the completion from the prefix.

Training streams endlessly (epochs reshuffle, with a `shuffle_buffer_size` buffer and `seed`), split across data
parallel ranks, and in each rank across the dataloader workers (`num_workers`), which make the examples on the fly:
texts are prefixed with BOS, split into words, and packed in order into examples of exactly `seq_length` words.
Longer texts are split into `seq_length`-word chunks, each its own sequence, and the rest is padded.
With the default pretokenizer, words longer than `max_word_length - 2` bytes are split.
Within the model, the bytes of all words (and the patches of rendered words) are packed without padding, so the
encoders and the bytes decoder only compute on real bytes. Resuming from a checkpoint restarts the stream.

### Validation

Each evaluation draws `validation.eval_iters` global batches. The validation examples (split across ranks) are
repeated to fill them, the same ones at every evaluation; if they do not fit, evaluations see only the first ones, and
a warning is logged: raise `eval_iters` to cover them.

## Parallelism

Data parallel only: `torchrun --nproc_per_node=N` (and Megatron's distributed optimizer). `train.global_batch_size`
must be a multiple of `micro_batch_size` × N; larger multiples accumulate gradients.
[`benchmarks/parity.sh`](../benchmarks/parity.sh) checks that 1 GPU and DP=2 train alike, on 2 GPUs.

## Optimizers

Megatron's optimizers, selected by `optimizer.optimizer`: `adam` (default), `sgd`, and the
[Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) `muon`, `adaptive_muon`, `soap`, `scion`,
`lion`, `polargrad`, ... Muon (2D weights orthogonalized, the rest with Adam) is used by
the task configs (e.g. [`string-repetition.yaml`](experiments/easy-tasks/string-repetition.yaml),
[`machine-translation.yaml`](experiments/machine-translation/machine-translation.yaml)); it improved every benchmarked
task over Adam (see [benchmarks](../benchmarks/README.md#tasks)).
Their hyperparameters are `OptimizerConfig` fields (e.g. `optimizer.muon_momentum`).

Only UTF-8 bytes are supported; the previous HuggingFace Trainer implementation is at the `huggingface-transformers`
git tag.

## Experimenting

- **A new task**: a YAML with `$extends: ../easy-tasks/string-repetition.yaml` (the path is relative to the new file),
  overriding the `data.dataset_*` keys, with a `[prefix, completion]` `dataset_text_template` so `welt-evaluate`
  works on it.
- **A new model size**: a HF config JSON in [`experiments/models`](experiments/models), referenced from `model.*`.

Where things are:
- [`welt/model.py`](../welt/model.py): the architecture.
- [`welt/processor.py`](../welt/processor.py): words to bytes, patches and labels, including shift-block masking.
- [`train.py`](train.py): config defaults, loss and metrics.
- [`data_utils.py`](data_utils.py): streaming and packing.
- [`welt/inference.py`](../welt/inference.py) and [`welt/server.py`](../welt/server.py): generation and serving.
- [`benchmarks/run_task.sh`](../benchmarks/run_task.sh): train, export, serve and evaluate a config.
