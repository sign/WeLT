from dataclasses import dataclass, field

from transformers.utils.versions import require_version


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_text_template: str | list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Template to format dataset text using Python format strings with dataset column names. "
                "Single string: concatenated text for training/eval (e.g., 'Translate: {source} to {target}'). "
                "List of 2 strings: [prefix, completion] for generation-based evaluation. "
                "  - During training: prefix + completion are concatenated. "
                "  - During evaluation/testing: prefix used for generation, completion used as reference. "
                "Example: ['<{sign_language}> {sign_text} <{spoken_language}> ', '{spoken_text}']"
            )
        },
    )
    train_file: str | None = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: str | None = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    max_sequence_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum sequence length for the model. "
                "Sequences will be truncated to this length if they are longer."
            )
        },
    )
    max_word_length: int | None = field(
        default=128,
        metadata={
            "help": (
                "Maximum word length for the model. "
                "Words will be truncated to this length if they are longer."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: int | None = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    metric_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "Evaluation metric(s) to compute (via evaluate.load()). "
                "For multiple metrics, use comma-separated list: 'bleu,chrf,meteor'. "
                "Used with generation-based evaluation when dataset_text_template is a list."
            )
        },
    )
    metric_for_best_model: str | None = field(
        default=None,
        metadata={
            "help": "Metric to use for selecting the best model checkpoint. Must be in metric_name if specified."
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        # Validate dataset_text_template
        if self.dataset_text_template is not None:
            if isinstance(self.dataset_text_template, list):
                if len(self.dataset_text_template) != 2:
                    msg = (
                        f"dataset_text_template must be either a string or a list of size 2. "
                        f"Got a list of size {len(self.dataset_text_template)}."
                    )
                    raise ValueError(msg)  # noqa: TRY003
            elif not isinstance(self.dataset_text_template, str):
                msg = (
                    f"dataset_text_template must be either a string or a list of size 2. "
                    f"Got {type(self.dataset_text_template).__name__}."
                )
                raise ValueError(msg)  # noqa: TRY003

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")  # noqa: TRY003
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
