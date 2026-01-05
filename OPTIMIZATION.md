We are now tasked, to autonomously improve the convergence of our string-repetition easy task benchmark.

To run a simple training experiment, run:
```shell
docker build -t welt . && \
docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)/welt:/app/welt" \
  -v "$(pwd)/welt_training:/app/welt_training" \
  -v /shared/.cache:/root/.cache \
  -v ~/.netrc:/root/.netrc:ro \
  -e WANDB_PROJECT="string-repetition" \
  -e WANDB_NAME="benchmark-run" \
  -e CONFIG="welt_training/experiments/easy-tasks/benchmark-run.yaml" \
  welt 
```
It trains for 500 steps, then performs a single evaluation pass on 32 samples.

The `tail -n 20` would be something like:
```log
***** eval metrics *****
  epoch                   =     0.1082
  eval_byte_accuracy      =     0.7652
  eval_chrf               =     8.6007
  eval_loss               =     0.7607
  eval_runtime            = 0:00:14.29
  eval_sacrebleu          =      0.301
  eval_samples            =         32
  eval_samples_per_second =      2.238
  eval_steps_per_second   =       0.07
  eval_word_accuracy      =     0.6277
  num_input_tokens_seen   =   32763904
  perplexity              =     2.1397
```

The easiest gains are likely from adjusting hyperparameters.
You may edit the configuration file `welt_training/experiments/easy-tasks/string-repetition-bench.yaml` to adjust them.

Our optimization task for today: we need to achieve the following metrics, in the least amount of clock time possible. 
- eval_loss < 0.76
- eval_sacrebleu > 0.3
- eval_byte_accuracy > 0.75

Run the initial setup as a benchmark. It should take about 12 minutes to complete on this machine.
Look at the logs, that look like:
> {'loss': 0.6916, 'grad_norm': 0.466796875, 'learning_rate': 4.259999999999999e-05, 'epoch': 0.09, 'num_input_tokens_seen': 28176384, 'train_runtime': 469.0089, 'train_tokens_per_second': 60076.44}

and based on the loss convergence, and `train_tokens_per_second` adjust the hyperparameters to improve convergence speed.

You can also profile the docker while it is running, to check memory use and GPU utilization, and make adjustments accordingly.

Hyperparameter adjustment alone should be sufficient to decrease from 12 minutes to under 5 minutes.

Once you are satisfied with your changes, you may also edit the code itself to further improve convergence speed.
This includes `welt_training/train.py` (the train loop), `welt_training/trainer.py` (the trainer, mostly used for eval),
and the model code under `welt/` such as `model.py` etc.

Progressively update "SPEEDRUN.md" with your changes and results as you go along, minimally, including:
- We observe something about the training dynamics
- We make a hypothesis about how to improve convergence speed
- The exact diff implemented
- The results observed (final metrics, time taken)

Start by reading SPEEDRUN.md for context on the current state of the optimization.

