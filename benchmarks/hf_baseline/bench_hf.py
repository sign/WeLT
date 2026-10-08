"""Steady-state throughput of the HF Trainer WeLT stack (the huggingface-transformers tag)."""
import json
import sys
import time

import welt_training.trainer as trainer_module
from transformers import TrainerCallback

START, END = 50, 300
result = {}

class Timer(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        import torch
        if state.global_step in (START, END):
            torch.cuda.synchronize()
            result[state.global_step] = time.perf_counter()
            if state.global_step == END:
                dt = (result[END] - result[START]) / (END - START)
                bs = args.per_device_train_batch_size
                out = {"stack": "hf", "config": sys.argv[1], "step_time_s": dt, "samples_per_s": bs / dt,
                       "word_positions_per_s": bs * 128 / dt,
                       "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
                print("BENCH", json.dumps(out), flush=True)

orig_init = trainer_module.WeLTTrainer.__init__
def init(self, *a, **k):
    orig_init(self, *a, **k)
    self.add_callback(Timer())
trainer_module.WeLTTrainer.__init__ = init

from welt_training.train import train  # noqa: E402

train(sys.argv[1])
