# Training

Training uses [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), inside the NeMo container
(see the [Dockerfile](../Dockerfile)).

```bash
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/easy-tasks/string-repetition.yaml
```

Any config value can be overridden from the command line, as `section.key=value`:

```bash
torchrun --nproc_per_node=8 -m welt_training.train welt_training/experiments/easy-tasks/string-repetition.yaml \
    output_dir=./output/test train.train_iters=50 train.global_batch_size=256 logger.wandb_project=null
```

## Config

```yaml
output_dir: ./output/string-repetition-tiny  # processor/, welt.yaml, checkpoints/, tensorboard/

model:
  bytes_encoder: sign/utf8-lm-tiny        # HF model id/path, or a JSON HF config, or null
  image_encoder: null                     # e.g. welt_training/experiments/models/image-encoder-tiny.json
  latent_transformer: sbintuitions/tiny-lm
  bytes_decoder: sign/utf8-lm-tiny
  load_pretrained: true                   # Initialize transformers from their HF weights
  # pretokenizer: EleutherAI/pythia-14m   # Defaults to sign/words-segmentation
  # modality_dropout: 0.15                # Any other WeLTModelProvider / TransformerConfig field

data:                                     # WeLTDatasetProvider fields
  dataset_name: Helsinki-NLP/opus-100
  dataset_config_name: en-he
  dataset_text_template: "<text>\x0E{translation[en]}\x0F<repeat> {translation[en]}"
  seq_length: 128                         # Words per packed example
  max_word_length: 16                     # Bytes per word, including BOS and EOS
  num_workers: 8

train:                                    # megatron.bridge TrainingConfig
  train_iters: 10000
  micro_batch_size: 32
  global_batch_size: 32
optimizer:                                # OptimizerConfig (distributed fused Adam by default)
  lr: 6.0e-4
scheduler:                                # SchedulerConfig (cosine by default)
  lr_warmup_iters: 500
validation:                               # ValidationConfig
  eval_interval: 100
  eval_iters: 1
checkpoint:                               # CheckpointConfig
  save_interval: 1000
logger:                                   # LoggerConfig
  wandb_project: string-repetition
```

Data can come from a HF dataset (`dataset_name`), local files (`train_file`, `validation_file`), or
shards made by `welt-prepare-data` (`prepared_data_path`). Large datasets can be `streaming: true`,
materializing `max_train_samples` and `max_eval_samples` examples.
Texts are pretokenized into words, and packed into examples of exactly `seq_length` words.

Within the model, the bytes of all words (and the patches of rendered words) are packed without padding, so
the encoders and the bytes decoder only compute on real bytes.

## Parallelism

Data parallelism (with the distributed optimizer) works out of the box with `torchrun --nproc_per_node=N`.
Pipeline and context parallelism are not supported.

## Not ported from the HuggingFace Trainer implementation

- UTF-16/UTF-32 encodings (`CharacterCausalLMWrapper`): only UTF-8 bytes are supported.
- `warmup_freeze_steps` and the Dion optimizer.
- Generation metrics during training: export a checkpoint and use [`welt.inference`](../welt/inference.py) instead.
- Pretrained HF image encoders (ViT, NaViT): image encoders are Megatron transformers over rendered word patches.
