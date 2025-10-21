# Chat

We would like to create a "ChatGPT Clone" using our new architecture, following
[nanochat](https://github.com/karpathy/nanochat).

Pretraining dataset: `karpathy/fineweb-edu-100b-shuffle`.
Midtraining dataset: `HuggingFaceTB/smoltalk2` (`Mid`).
SFT dataset: `HuggingFaceTB/smoltalk2` (`SFT`).
RL dataset: `HuggingFaceTB/smoltalk2` (`Preference`).

## Base model (pretraining)

We train a model on `karpathy/fineweb-edu-100b-shuffle` -
Chinchilla says #tokens = 20X #params, so we need 10B/20 = 500m parameters.

```shell
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

## Midtraining

Teach the model conversation special tokens, tool use, multiple choice.

```shell
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

## Supervised Finetuning

Domain adaptation to each sequence all by itself per row (no packing)
train sft and re-eval right away (should see a small bump)

```shell
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

## Reinforcement Learning

```shell
# run reinforcement learning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

## Chat Interface

https://github.com/karpathy/nanochat/blob/master/scripts/chat_web.py