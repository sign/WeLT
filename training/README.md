# Training

> [!CAUTION]
> Read the [vision README](../welt/vision/README.md) to understand how to select
> an image encoder model for fast training.

Setup with:

```bash
pip install ".[train]"
```

Run:
[//]: # (TODO: Unclear why `remove_unused_columns=False` is needed, but it is required to avoid errors during training.)

```bash
python -m training.train \
    --image_encoder_model_name_or_path "WinKawaks/vit-tiny-patch16-224" \
    --bytes_encoder_model_name_or_path "prajjwal1/bert-tiny" \
    --latent_transformer_model_name_or_path "sbintuitions/tiny-lm" \
    --bytes_decoder_model_name_or_path "sbintuitions/tiny-lm" \
    --load_pretrained True \
    --dataset_name Helsinki-NLP/opus-100 \
    --dataset_config_name en-he \
    --dataset_text_template "<en> {translation[en]} <he> {translation[he]}" \
    --remove_unused_columns False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --save_steps 10 \
    --output_dir output \
    --overwrite_output_dir \
    --logging_steps 1 \
    --logging_strategy steps \
    --max_steps 50 \
    --max_sequence_length 32 \
    --max_word_length 8 \
    --dataloader_num_workers 4 \
    --include_tokens_per_second True \
    --include_num_input_tokens_seen True \
    --max_train_samples 16 \
    --warmup_freeze_steps 10
```

Use `warmup_freeze_steps=N` to freeze the pretrained modules for the first N steps
([#7](https://github.com/sign/WeLT/issues/7)).

### Training Quirks

`num_input_tokens_seen` and `train_tokens_per_second` are calculated based on the number of bytes the model decodes.
That means that in practice, if `max_word_length=32`, a rough estimate of
the real number of **words** the model sees should be divided by 32.

### Performance Optimization

To speed up the processor's image rendering and preprocessing, you can
increase `processor.cache_size` to cache more preprocessed images in memory.
(`cache_size=500_000` can take 25GB of RAM per process).
Ideally, we make the renderer so fast it doesn't need caching at all.