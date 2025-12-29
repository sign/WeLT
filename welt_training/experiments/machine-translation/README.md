# Machine Translation

We compare HuggingFace's example training for a causal language model to our setup.

We modified the `run_clm.py` script, in this directory.
See the modifications by running a diff with: 
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

We train `EleutherAI/pythia-70m` from scratch, using:

```shell
python run_clm.py \
    --model_name_or_path EleutherAI/pythia-70m \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --block_size 128 \
    --output_dir ./output/clm \
```

Compared to our model setup:
```shell
welt-train \
    --image_encoder_model_name_or_path WinKawaks/vit-tiny-patch16-224 \
    --bytes_encoder_model_name_or_path prajjwal1/bert-tiny \
    --latent_transformer_model_name_or_path EleutherAI/pythia-70m \
    --bytes_decoder_model_name_or_path sbintuitions/tiny-lm \
    --remove_unused_columns False \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_sequence_length 128 \
    --max_word_length 20 \
    --output_dir ./output/welt \
    --dtype bfloat16 \
```

With the following shared arguments:

```shell
    --dataset_name Helsinki-NLP/opus-100 \
    --dataset_config_name en-he \
    --dataset_text_template "<en> {translation[en]} <he> {translation[he]}" \
    --do_train True \
    --do_eval True \
    --metric_for_best_model accuracy \
    --eval_on_start True \
    --eval_strategy epoch \
    --logging_steps 10 \
    --logging_strategy steps \
    --max_steps 100000 \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --auto_find_batch_size True \
    --include_tokens_per_second True \
    --include_num_input_tokens_seen True \
    --learning_rate 3e-4 \
    --optim adamw_torch_fused \
    --bf16 True \
    --report_to wandb
```

### Using configs

```bash
welt-train welt_training/experiments/machine-translation/machine-translation.yaml
welt-train welt_training/experiments/machine-translation/machine-translation-signed-spoken.yaml
```