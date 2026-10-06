"""
Training arguments for WeLT Trainer.

Extends Seq2SeqTrainingArguments with additional parameters for generation-based evaluation.
"""
import math
from dataclasses import dataclass, field

from transformers import Seq2SeqTrainingArguments
from transformers.training_args import OptimizerNames

# Monkey-patch OptimizerNames to accept "dion"
OptimizerNames._value2member_map_["dion"] = "dion"  # type: ignore[attr-defined]


@dataclass
class WeLTTrainingArguments(Seq2SeqTrainingArguments):
    """
    Training arguments for WeLT Trainer.

    Extends Seq2SeqTrainingArguments with parameters specific to WeLT's
    generation-based evaluation.
    """

    # Keep existing experiment YAMLs usable after Transformers 5 removed these fields.
    overwrite_output_dir: bool = field(default=False)
    include_tokens_per_second: bool = field(default=False)
    warmup_ratio: float = field(default=0.0)

    def get_warmup_steps(self, num_training_steps: int):
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.warmup_steps == 0 and self.warmup_ratio > 0:
            return math.ceil(num_training_steps * self.warmup_ratio)
        return super().get_warmup_steps(num_training_steps)

    eval_metrics: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "List of evaluation metrics to compute during generation-based evaluation. "
                "Examples: ['sacrebleu', 'chrf', 'bleu', 'rouge']. "
                "If None, only eval_loss, byte_accuracy, word_accuracy, and perplexity will be computed."
            )
        },
    )

    log_samples: int = field(
        default=3,
        metadata={
            "help": (
                "Number of sample predictions to log during evaluation. "
                "Set to 0 to disable sample logging."
            )
        },
    )

    # FLOPS profiling options
    profile_flops: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable FLOPS profiling to measure GPU utilization and TFLOPS/s by dtype. "
                "Uses PyTorch profiler to capture CUDA operations."
            )
        },
    )

    flops_profile_steps: int = field(
        default=50,
        metadata={
            "help": "Profile FLOPS every N training steps."
        },
    )

    flops_warmup_steps: int = field(
        default=2,
        metadata={
            "help": "Number of warmup steps before recording profiler data."
        },
    )

    flops_active_steps: int = field(
        default=3,
        metadata={
            "help": "Number of active profiling steps to record."
        },
    )
