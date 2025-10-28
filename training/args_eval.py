from dataclasses import dataclass, field

from transformers.utils.versions import require_version


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

   
    eval_metrics: list[str] = field(
        default_factory = lambda: ["accuracy"],
        metadata={
            "help": (
                "Names of the metrics to evalulate the data on. Should be a list of string"
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
    

    def __post_init__(self):
        if self.eval_metrics:
            for em in self.eval_metrics:
                assert em in ["accuracy", "cer", "wer", "bleu", "rouge"], f"{em} should be in ['accuracy', 'cer', 'wer', 'bleu', 'rouge']"
