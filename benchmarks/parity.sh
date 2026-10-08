#!/bin/bash
# Parallelism parity on 2 GPUs: the same global batches on 1 GPU, with data parallelism (DP=2),
# and with tensor + sequence parallelism (TP=2). Losses should match closely. Then exports the TP=2 checkpoint.
#   bash benchmarks/parity.sh [output_dir]
out=${1:-./output/parity}
common="welt_training/experiments/machine-translation/machine-translation.yaml train.train_iters=200
  train.micro_batch_size=32 train.global_batch_size=64 validation.eval_interval=100 validation.eval_iters=2
  checkpoint.save_interval=200 logger.wandb_project=null logger.log_interval=20 model.modality_dropout=0.0
  data.max_train_samples=100000"
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m welt_training.train $common output_dir=$out/gpu1 > $out/gpu1.log 2>&1
torchrun --nproc_per_node=2 -m welt_training.train $common output_dir=$out/dp2 > $out/dp2.log 2>&1
torchrun --nproc_per_node=2 -m welt_training.train $common output_dir=$out/tp2 \
  model.tensor_model_parallel_size=2 model.sequence_parallel=true > $out/tp2.log 2>&1
for run in gpu1 dp2 tp2; do
  echo "== $run"
  grep -oE "iteration +[0-9]+/|per iteration \(ms\): [0-9.]+|lm loss: [0-9.E+-]+|lm loss value: [0-9.E+-]+" $out/$run.log | paste -sd' '
done
CUDA_VISIBLE_DEVICES=0 welt-export $out/tp2/checkpoints --output $out/tp2-export | grep Exported
