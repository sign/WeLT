# Training

```shell
torchrun --nproc_per_node=<gpus> -m welt_training.train <config.yaml> [section.key=value ...]
```

Overrides are YAML values, at any depth: `output_dir=./output/test`, `train.train_iters=50`,
`logger.wandb_project=null`, `model.modality_dropout=0.3`.
Configs can inherit with `$extends: ./other.yaml` (relative to the file), deep-merging their sections.

## Config

```yaml
output_dir: ./output/string-repetition-tiny  # welt.yaml, processor/, checkpoints/, tensorboard/

model:
  bytes_encoder: sign/utf8-lm-tiny        # HF model id/path, a JSON HF config, or null
  image_encoder: null                     # A JSON HF config (patch transformer), a HF vision backbone, or null
  latent_transformer: sbintuitions/tiny-lm
  bytes_decoder: sign/utf8-lm-tiny
  load_pretrained: true                   # Initialize id/path transformers from their HF weights
  # pretokenizer: EleutherAI/pythia-14m   # A HF tokenizer splitting words, defaults to sign/words-segmentation
  # trust_remote_code: false
  # modality_dropout: 0.15                # Any other WeLTModelProvider field, e.g. tensor_model_parallel_size

data:
  dataset_name: Helsinki-NLP/opus-100
  dataset_config_name: en-he
  dataset_text_template:                  # [prefix, completion] (concatenated for training), or one string
    - "<text>\x0E{translation[en]}\x0F<repeat> "
    - "{translation[en]}"
  seq_length: 128                         # Words per packed example
  max_word_length: 16                     # Bytes per word, including BOS and EOS
  max_eval_samples: 32
  num_workers: 8

# Megatron-Bridge sections, keys set as-is on its configs (unknown keys raise)
train:                                    # TrainingConfig
  train_iters: 10000
  micro_batch_size: 32
  global_batch_size: 32
optimizer:                                # OptimizerConfig
  lr: 6.0e-4
  min_lr: 6.0e-5
scheduler:                                # SchedulerConfig
  lr_warmup_iters: 500
validation:                               # ValidationConfig
  eval_interval: 100
  eval_iters: 1
checkpoint:                               # CheckpointConfig
  save_interval: 1000
logger:                                   # LoggerConfig
  wandb_project: string-repetition        # null disables W&B; wandb_exp_name names the run
# ddp: DistributedDataParallelConfig, rng: RNGConfig
```

Defaults ([`train.py`](train.py) `build_config`): 1000 iterations of batch 32, distributed Adam
(lr 3e-4 → 3e-5 cosine, betas 0.9/0.95, weight decay 0.01, grad clip 1.0), bf16, evaluation every 500 iterations,
checkpoints every 1000 to `<output_dir>/checkpoints` (also loaded from there, to resume), TensorBoard logs to
`<output_dir>/tensorboard`, seed 42.

### Model

`model` keys other than the four transformers, `load_pretrained`, `pretokenizer` and `trust_remote_code` are set on
the `WeLTModelProvider` ([`welt/model.py`](../welt/model.py)), the latent transformer's Megatron config, e.g.
`modality_dropout` (default 0.15: with both encoders, each one's embeddings are dropped with this probability per
step, rescaling the other), `tensor_model_parallel_size`, `sequence_parallel`, or `recompute_granularity`.
Parallelism and recomputation settings are shared with the other transformers.

Image encoders:
- **Patch transformer**: a causal LM config or model (e.g. [`models/image-encoder-tiny.json`](experiments/models/image-encoder-tiny.json)),
  used bidirectionally over the 16x16 patches of each word, built by Megatron. See [`ocr.yaml`](experiments/easy-tasks/ocr.yaml).
- **HF vision backbone** (ViT, DeiT, DINOv2, CLIP, SigLIP, ...: any config with a `patch_size`), run with
  transformers, pretrained with `load_pretrained: true`. See [`ocr-vit.yaml`](experiments/easy-tasks/ocr-vit.yaml).

### Data

The `data` section configures the `WeLTDatasetProvider` ([`data.py`](data.py), [`data_utils.py`](data_utils.py)).
Texts come from one of:
- a HF dataset: `dataset_name`, `dataset_config_name`. Without a validation split, `validation_split_percentage`
  (default 5) of train is held out. Large datasets can be `streaming: true`, which materializes the first
  `max_train_samples` and `max_eval_samples` examples (both required).
- local files: `train_file`, `validation_file` (`.txt`, `.json`, `.csv`, ...).
- shards made by `welt-prepare-data`: `prepared_data_path` (see the [README](../README.md#data-preparation)).

`dataset_text_template` is a Python format string over the dataset's columns (without it, the `text` column, or the
first). As `[prefix, completion]`, both parts are concatenated for training, and `welt-evaluate` generates the
completion from the prefix. `max_train_samples` / `max_eval_samples` cap the number of texts per split.

Texts are prefixed with BOS, split into words, and packed in order into examples of exactly `seq_length` words
(longer texts are truncated, the rest padded). With the default pretokenizer, words longer than
`max_word_length - 2` bytes are split.
Within the model, the bytes of all words (and the patches of rendered words) are packed without padding, so the
encoders and the bytes decoder only compute on real bytes.
Dataloader options (`num_workers`, `pin_memory`, ...) and `preprocessing_num_workers` also go in `data`.

### Validation

Each evaluation draws `validation.eval_iters × train.global_batch_size` packed examples. When that is at least the
number of packed validation examples, every evaluation covers the whole validation set (examples repeat to fill
the batches); otherwise only part of it is, and a warning is logged. Use `max_eval_samples` to keep
the validation set small, and raise `eval_iters` to cover it.

## Parallelism

- **Data parallel**: `torchrun --nproc_per_node=N` (and Megatron's distributed optimizer). `train.global_batch_size`
  must be a multiple of `micro_batch_size` × the data parallel size; larger multiples accumulate gradients.
- **Tensor parallel**: `model.tensor_model_parallel_size=T`, optionally with `model.sequence_parallel=true`.
  Data parallel size is then `N / T`.
- Pipeline and context parallelism are not supported.

## Optimizers

Megatron's optimizers, selected by `optimizer.optimizer`: `adam` (default), `sgd`, `lion`,
`muon` (2D weights orthogonalized, the rest with Adam; see [`letter-count.yaml`](experiments/easy-tasks/letter-count.yaml)),
`adaptive_muon`, `soap`, `scion`, `psgd_pro` and the other
[Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers).
Their hyperparameters are `OptimizerConfig` fields (e.g. `optimizer.muon_momentum`).

## Not ported from the HuggingFace Trainer implementation

The previous implementation is kept at the `huggingface-transformers` git tag. Not ported:
- UTF-16/UTF-32 encodings (`CharacterCausalLMWrapper`): only UTF-8 bytes are supported.
- `warmup_freeze_steps` and the Dion optimizer.
- Generation metrics during training: export a checkpoint and run `welt-evaluate` instead.
