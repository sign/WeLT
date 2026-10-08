# Training

```shell
torchrun --nproc_per_node=<gpus> -m welt_training.train <config.yaml> [section.key=value ...]
```

Overrides are YAML values, at any depth: `output_dir=./output/test`, `train.train_iters=50`,
`logger.wandb_project=null` (to run without W&B), `model.modality_dropout=0.3`.
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
  modality_dropout: ...   # Any other WeLTModelProvider field, e.g. tensor_model_parallel_size
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
`modality_dropout` (default 0.15: with both encoders, each one's embeddings are dropped with this probability per
step, rescaling the other), `tensor_model_parallel_size`, `sequence_parallel`, or `recompute_granularity`.
Parallelism, precision and recomputation settings are shared with the other transformers
(`SHARED_CONFIG_FIELDS`); other fields, e.g. `hidden_dropout`, only apply to the latent transformer.

For the image encoder (described in the [README](../README.md#model-setup)), see
[`ocr.yaml`](experiments/easy-tasks/ocr.yaml) and [`image-encoder-tiny.json`](experiments/models/image-encoder-tiny.json).

### Data

The `data` section configures the `WeLTDatasetProvider` ([`data.py`](data.py), [`data_utils.py`](data_utils.py)).
Texts come from one of:
- a HF dataset: `dataset_name`, `dataset_config_name`. Large datasets can be `streaming: true`, which materializes
  the first `max_train_samples` and `max_eval_samples` examples (both required).
- local files: `train_file`, `validation_file` (`.txt`, `.json`, `.csv`, ...).
- shards made by `welt-prepare-data`: `prepared_data_path` (see [Data Preparation](#data-preparation)).

Without a validation split, `validation_split_percentage` (default 5) of train is held out (when streaming, the first
`max_eval_samples` examples of train).

`dataset_text_template` is a Python format string over the dataset's columns; without it, the `text` column is used
(a missing `text` column raises). As `[prefix, completion]`, both parts are concatenated for training, and
`welt-evaluate` generates the completion from the prefix. `max_train_samples` / `max_eval_samples` cap the number of
texts per split.

Texts are prefixed with BOS, split into words, and packed in order into examples of exactly `seq_length` words:
longer texts are split into `seq_length`-word chunks, each its own sequence, and the rest is padded.
With the default pretokenizer, words longer than `max_word_length - 2` bytes are split.
Within the model, the bytes of all words (and the patches of rendered words) are packed without padding, so the
encoders and the bytes decoder only compute on real bytes.
Dataloader options (`num_workers`, `pin_memory`, ...) and `preprocessing_num_workers` also go in `data`.

#### Data Preparation

For large-scale training, `welt-prepare-data` streams a HuggingFace dataset (shuffled), chunks documents into
examples of words, and writes sharded `.jsonl.gz` files; several datasets can share one directory.
Match the training config: `--max_seq_length` is `data.seq_length - 1` (training adds a BOS word), and
`--max_bytes_per_word` is `data.max_word_length - 2`.

```shell
welt-prepare-data \
    --dataset_name HuggingFaceFW/fineweb --dataset_config sample-10BT --language eng_Latn \
    --train_split_units 3200000000 --validation_split_units 100000000 --num_units_per_file 100000000 \
    --max_seq_length 511 --max_bytes_per_word 30 \
    --output_path /scratch/data/pretrain
welt-verify-data --data_path /scratch/data/pretrain
torchrun --nproc_per_node=8 -m welt_training.train welt_training/experiments/pretrain/pile-pretrain-70m-no-image.yaml
```

This writes `{dataset}-{config}-{split}-{index}.jsonl.gz` shards and a `{dataset}-{config}-{split}-metadata.json`
per split. Units are words (or `--unit_type chars`); validation is filled first, then train.
`welt-verify-data` checks shard and example counts against the metadata, and warns when train and validation of the
same source were prepared separately (risking overlap). See `welt-prepare-data --help` for all options
(`--text_column`, `--text_template`, `--id_column`, `--drop_remainder`, `--seed`, ...).

### Validation

Each evaluation draws `validation.eval_iters × train.global_batch_size` packed examples. When that is at least the
number of packed validation examples, every evaluation covers the whole validation set (examples repeat to fill
the batches); otherwise only part of it is, and a warning is logged. Use `max_eval_samples` to keep
the validation set small, and raise `eval_iters` to cover it.

## Parallelism

- **Data parallel**: `torchrun --nproc_per_node=N` (and Megatron's distributed optimizer). `train.global_batch_size`
  must be a multiple of `micro_batch_size` × the data parallel size; larger multiples accumulate gradients.
- **Tensor parallel**: `model.tensor_model_parallel_size=T` with `model.sequence_parallel=true` (required).
  T must divide `seq_length`, each transformer's attention heads and query groups, and 256 (the byte vocabulary).
  Data parallel size is then `N / T`.
- Pipeline and context parallelism are not supported.

[`benchmarks/parity.sh`](../benchmarks/parity.sh) checks that 1 GPU, DP=2 and TP=2 (with sequence parallelism)
train alike, on 2 GPUs.

## Optimizers

Megatron's optimizers, selected by `optimizer.optimizer`: `adam` (default), `sgd`, and the
[Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) `muon`, `adaptive_muon`, `soap`, `scion`,
`lion`, `polargrad`, ... Muon (2D weights orthogonalized, the rest with Adam) is used by
[`letter-count.yaml`](experiments/easy-tasks/letter-count.yaml) and [`single-query.yaml`](experiments/chat/single-query.yaml).
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
- [`data_utils.py`](data_utils.py): loading and packing.
- [`welt/inference.py`](../welt/inference.py) and [`welt/server.py`](../welt/server.py): generation and serving.
- [`benchmarks/run_task.sh`](../benchmarks/run_task.sh): train, export, serve and evaluate a config.
