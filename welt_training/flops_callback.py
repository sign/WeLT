"""
FLOPS tracking callback for WeLT training.

Uses PyTorch profiler to measure actual FLOPS during training steps.
"""
import logging

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class FlopsCallback(TrainerCallback):
    """
    Callback that profiles training steps and reports FLOPS.

    Uses torch.profiler to measure actual operations and their FLOPS.
    Works on both CUDA and CPU.
    """

    def __init__(
        self,
        profile_steps: int = 50,
        warmup_steps: int = 2,
        active_steps: int = 3,
    ):
        """
        Initialize FlopsCallback.

        Args:
            profile_steps: Profile every N steps (default: 50)
            warmup_steps: Profiler warmup steps before recording (default: 2)
            active_steps: Number of steps to actively profile (default: 3)
        """
        self.profile_steps = profile_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps

        self._profiler = None
        self._profiler_step = 0
        self._profile_start_step = None

        print(f"FlopsCallback initialized: profile every {profile_steps} steps, "
              f"{warmup_steps} warmup + {active_steps} active steps per profile")

    def _should_start_profiling(self, global_step: int) -> bool:
        """Check if we should start a new profiling session."""
        return (
            self._profiler is None
            and global_step > 0
            and global_step % self.profile_steps == 0
        )

    def _get_profiler_activities(self):
        """Get profiler activities based on available hardware."""
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        return activities

    def _start_profiler(self, global_step: int):
        """Start a new profiling session."""
        self._profile_start_step = global_step
        self._profiler_step = 0

        self._profiler = profile(
            activities=self._get_profiler_activities(),
            schedule=schedule(
                wait=0,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=1,
            ),
            with_flops=True,
            record_shapes=True,
            profile_memory=False,
        )
        self._profiler.__enter__()

    def _stop_profiler_and_report(self, global_step: int, trainer):
        """Stop profiler, analyze results, and report metrics."""
        if self._profiler is None:
            return

        try:
            self._profiler.__exit__(None, None, None)
            events = self._profiler.key_averages()
            metrics = self._analyze_flops(events)
            self._log_to_console(metrics, global_step)

            if trainer is not None and metrics:
                trainer.log(metrics)

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error analyzing profiler results: {e}")
        finally:
            self._profiler = None
            self._profile_start_step = None

    def _analyze_flops(self, events) -> dict:
        """Analyze profiler events and extract FLOPS metrics and time breakdown."""
        metrics = {}

        total_flops = 0
        total_cuda_time_us = 0
        total_cpu_time_us = 0

        # Track ALL operations by time (not just ones with FLOPS)
        all_ops = []

        for event in events:
            # Skip profiler overhead
            if 'ProfilerStep' in event.key:
                continue

            cpu_time = event.self_cpu_time_total
            cuda_time = event.self_device_time_total  # "Self CUDA" in profiler table

            total_cpu_time_us += cpu_time
            total_cuda_time_us += cuda_time

            op_info = {
                'name': event.key,
                'flops': event.flops or 0,
                'cpu_time_us': cpu_time,
                'cuda_time_us': cuda_time,
            }
            all_ops.append(op_info)

            if event.flops and event.flops > 0:
                total_flops += event.flops

        # Sort by CUDA time (where time is actually spent)
        all_ops.sort(key=lambda x: x['cuda_time_us'], reverse=True)
        self._top_ops_by_time = all_ops[:15]

        # Calculate metrics
        metrics['flops/total_gflops'] = round(total_flops / 1e9, 2)
        metrics['flops/total_cuda_time_ms'] = round(total_cuda_time_us / 1e3, 2)
        metrics['flops/total_cpu_time_ms'] = round(total_cpu_time_us / 1e3, 2)

        if total_cuda_time_us > 0:
            time_s = total_cuda_time_us / 1e6
            metrics['flops/tflops_per_sec'] = round((total_flops / 1e12) / time_s, 2)

        return metrics

    def _log_to_console(self, metrics: dict, global_step: int):
        """Log FLOPS metrics to console."""
        if not metrics:
            return

        tflops = metrics.get('flops/tflops_per_sec', 0)
        total_gflops = metrics.get('flops/total_gflops', 0)
        cuda_time_ms = metrics.get('flops/total_cuda_time_ms', 0)
        cpu_time_ms = metrics.get('flops/total_cpu_time_ms', 0)

        print(f"\n{'='*70}")
        print(f"FLOPS Profile @ Step {global_step}")
        print(f"{'='*70}")
        print(f"  Compute: {total_gflops:.2f} GFLOPS | {tflops:.2f} TFLOPS/s")
        print(f"  Time:    CUDA {cuda_time_ms:.2f}ms | CPU {cpu_time_ms:.2f}ms")

        # Show where CUDA time is spent (the key insight)
        if hasattr(self, '_top_ops_by_time') and self._top_ops_by_time:
            print("\n  Top Operations by CUDA Time:")
            for i, op in enumerate(self._top_ops_by_time, 1):
                cuda_ms = op['cuda_time_us'] / 1e3
                pct = (op['cuda_time_us'] / (cuda_time_ms * 1e3) * 100) if cuda_time_ms > 0 else 0
                gflops = op['flops'] / 1e9 if op['flops'] else 0
                flops_str = f"{gflops:7.1f} GFLOPS" if gflops > 0 else "       -      "
                print(f"    {i:2d}. {op['name'][:40]:40s} {cuda_ms:8.3f}ms ({pct:5.1f}%) {flops_str}")

        print(f"{'='*70}\n")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, **kwargs):
        if self._should_start_profiling(state.global_step):
            self._start_profiler(state.global_step)
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        if self._profiler is not None:
            self._profiler.step()
            self._profiler_step += 1

            total_profiler_steps = self.warmup_steps + self.active_steps
            if self._profiler_step >= total_profiler_steps:
                trainer = kwargs.get('trainer')
                self._stop_profiler_and_report(state.global_step, trainer)

        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        if self._profiler is not None:
            trainer = kwargs.get('trainer')
            self._stop_profiler_and_report(state.global_step, trainer)
        return control
