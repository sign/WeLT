from dataclasses import dataclass, field

from transformers import Seq2SeqTrainingArguments


@dataclass
class EvaluationArguments(Seq2SeqTrainingArguments):
    """
    Extra arguments for evaluation on top of Seq2SeqTrainingArguments.
    """

    eval_metrics: list[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={
            "help": (
                "Names of the metrics to evaluate the data on. Should be a list of strings "
                "e.g. ['bleu', 'cer']"
            )
        },
    )

    max_train_samples_for_eval: int | None = field(
        default=None,
        metadata={
            "help": (
                "The number of samples from training set to use to compute evaluation metrics. "
                "Defaults to all if not set."
            )
        },
    )
    max_eval_samples_for_eval: int | None = field(
        default=None,
        metadata={
            "help": (
                "The number of samples from validation set to use to compute evaluation metrics. "
                "Defaults to all if not set."
            )
        },
    )
    max_test_samples_for_eval: int | None = field(
        default=None,
        metadata={
            "help": (
                "The number of samples from test set to use to compute evaluation metrics. "
                "Defaults to all if not set."
            )
        },
    )

    log_examples_every: int | None = field(
        default=None,
        metadata={"help": "If set, logs example predictions every N batches."},
    )

    # Your legacy batch_size – we map it to per_device_eval_batch_size
    batch_size: int = field(
        default=1024,
        metadata={"help": "Batch size for evaluation (mapped to per_device_eval_batch_size)."},
    )

    def __post_init__(self):
        super().__post_init__()

        viable_eval_metrics = ["accuracy", "cer", "wer", "bleu", "rouge", "sacrebleu"]
        if self.eval_metrics:
            for em in self.eval_metrics:
                assert em in viable_eval_metrics, f"{em} should be in {viable_eval_metrics}"

        if self.batch_size is not None and self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.batch_size

        if not getattr(self, "predict_with_generate", False):
            self.predict_with_generate = True

        if self.generation_max_length is None:
            self.generation_max_length = 8
