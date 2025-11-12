from dataclasses import dataclass, field


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """


    task_word: str = field(
        default = None,
        metadata={
            "help": (
                "The task seperator enclosed in <>"
            )
        },
    )
    eval_metrics: list[str] = field(
        default_factory = lambda: ["accuracy"],
        metadata={
            "help": (
                "Names of the metrics to evaluate the data on. Should be a list of string"
            )
        },
    )
    max_train_samples_for_eval: int | None = field(
        default=None,
        metadata={
            "help": (
                "The number of samples from training set to use to compute evaluation metrics. Defaults to"
                "all if not set"
            )
        },
    )
    max_eval_samples_for_eval: int | None = field(
        default=None,
        metadata={
            "help": (
                "The number of samples from validation set to use to compute evaluation metrics. Defaults to"
                "all if not set"
            )
        },
    )
    max_test_samples_for_eval: int | None = field(
        default=None,
        metadata={
            "help": (
                "The number of samples from test set to use to compute evaluation metrics. Defaults to"
                "all if not set"
            )
        },
    )

    log_examples_every: int | None = field(
        default=None,
        metadata={
            "help": (
                "If set, logs example predictions every N batches."
            )
        },
    )
    batch_size: int = field(
        default=1024,
        metadata={
            "help": (
                "Batch size for evaluation."
            )
        },
    )




    def __post_init__(self):
        EVAL_METRICS = ["accuracy", "cer", "wer", "bleu", "rouge", "sacrebleu"]
        if self.eval_metrics:
            for em in self.eval_metrics:
                assert em in EVAL_METRICS, f"{em} should be in {EVAL_METRICS}"
