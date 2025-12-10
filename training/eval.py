"""
Evaluation script using WeLTTrainer for generation-based metrics.

This script evaluates a trained model on train, validation, and test splits
using the WeLTTrainer's generation-based evaluation loop.
"""

import json
import logging
import os
import sys
from pathlib import Path

from transformers import GenerationConfig, HfArgumentParser
from transformers.utils import send_example_telemetry

from training.args_data import DataTrainingArguments
from training.args_eval import EvaluationArguments
from training.args_model import ModelArguments
from training.train import init_model, limit_dataset_size, load_dataset, split_streaming_dataset
from welt.trainer import WeLTTrainer

logger = logging.getLogger(__name__)


def init_datasets(data_args: DataTrainingArguments,  # noqa: C901
                  trust_remote_code: bool,
                  do_train: bool = True,
                  cache_dir: str = None):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=cache_dir,
            streaming=data_args.streaming,
            trust_remote_code=trust_remote_code,
        )
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split="train",
                    cache_dir=cache_dir,
                    streaming=data_args.streaming,
                    trust_remote_code=trust_remote_code,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=cache_dir,
                    streaming=data_args.streaming,
                    trust_remote_code=trust_remote_code,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=cache_dir,
                    streaming=data_args.streaming,
                    trust_remote_code=trust_remote_code,
                )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=cache_dir,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    extension,
                    data_files=data_files,
                    split="train",
                    cache_dir=cache_dir,
                    **dataset_args,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=cache_dir,
                    **dataset_args,
                )

                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=cache_dir,
                    **dataset_args,
                )

    if do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def process_split(dataset, split_name: str):
        """Apply mapping and filtering to a dataset split."""
        template = data_args.dataset_text_template
        if template is None:
            def mapping_fn(example):
                return {"text": example[text_column_name]}
        else:
            is_single_text_template = isinstance(template, str)
            single_text_template = template if is_single_text_template else "".join(template)

            # Only treat "train" specially when we are actually training
            is_train_like = (split_name == "train") and do_train

            def mapping_fn(example):
                # During training: single "text" field for train split.
                # During pure eval (do_train=False): even "train" gets prefix/completion.
                if is_single_text_template or is_train_like:
                    return {"text": single_text_template.format(**example)}

                prefix = template[0].format(**example)
                completion = template[1].format(**example)
                return {
                    "text": prefix,                   # Will be using this as input for prediction
                    "prefix": prefix,                 # For generation
                    "completion": completion,         # Reference for metrics
                }

        map_args = {}
        if not data_args.streaming:
            map_args = {
                "num_proc": data_args.preprocessing_num_workers,
                "load_from_cache_file": not data_args.overwrite_cache,
            }

        dataset = dataset.map(
            mapping_fn,
            remove_columns=column_names,
            desc=f"Formatting {split_name} split",
            **map_args
        )
        dataset = dataset.filter(
            lambda x: len(x["text"]) > 0,
            desc=f"Filtering empty examples from {split_name}",
            **map_args
        )
        return dataset

    return {split: process_split(raw_datasets[split], split) for split in raw_datasets}


def init_logging(eval_args: EvaluationArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    logger.info(f"Evaluation arguments: {eval_args}")


def parse_args_into_dataclasses(args: list[str] | None | str = None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvaluationArguments))
    # If we pass only one argument to the script and it's the path to a json or yaml file,
    # let's parse it to get our arguments.
    if isinstance(args, str):
        return parser.parse_yaml_file(yaml_file=os.path.abspath(args))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses(args=args)


def write_predictions(
    predictions: list[str],
    completions: list[str],
    file_path: Path,
    split_name: str,
):
    """Write predictions and gold completions to a JSONL file."""
    with open(file_path, "w") as f:
        for ref, hyp in zip(completions, predictions, strict=False):
            f.write(json.dumps({"reference": ref, "prediction": hyp}) + "\n")
    logger.info(f"{split_name} predictions written to {file_path}")


def eval(args: list[str] | None | str = None):  # noqa: C901
    """Run evaluation on train, validation, and test splits."""

    model_args, data_args, eval_args = parse_args_into_dataclasses(args)

    init_logging(eval_args)

    send_example_telemetry("run_clm", model_args, data_args)

    # Initialize model, processor, and collator
    model, processor, collator = init_model(model_args, data_args, seed=42)

    # Load datasets
    text_datasets = init_datasets(
        data_args,
        cache_dir=None,
        trust_remote_code=model_args.trust_remote_code,
        do_train=False,
    )

    # Prepare datasets for each split
    if "train" not in text_datasets:
        raise ValueError("--do_train requires a train dataset")  # noqa: TRY003
    train_dataset = limit_dataset_size(
        text_datasets["train"],
        max_samples=eval_args.max_train_samples_for_eval,
        streaming=data_args.streaming,
    )

    if "validation" not in text_datasets:
        raise ValueError("--do_eval requires a validation dataset")  # noqa: TRY003
    eval_dataset = limit_dataset_size(
        text_datasets["validation"],
        max_samples=eval_args.max_eval_samples_for_eval,
        streaming=data_args.streaming,
    )

    if "test" not in text_datasets:
        raise ValueError("--do_eval requires a test dataset")  # noqa: TRY003
    test_dataset = limit_dataset_size(
        text_datasets["test"],
        max_samples=eval_args.max_test_samples_for_eval,
        streaming=data_args.streaming,
    )

    # Apply processor transform to datasets
    train_dataset = train_dataset.with_transform(processor)
    eval_dataset = eval_dataset.with_transform(processor)
    test_dataset = test_dataset.with_transform(processor)

    # Set up generation config
    bytes_generation_config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
    )

    # Set up output directory
    model_dir = (
        Path(model_args.model_name_or_path).parent
        if model_args.model_name_or_path.endswith(".safetensors")
        else Path(model_args.model_name_or_path)
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    # File paths for predictions
    train_pred_file_path = model_dir / "train_predictions.jsonl"
    eval_pred_file_path = model_dir / "eval_predictions.jsonl"
    test_pred_file_path = model_dir / "test_predictions.jsonl"

    # Create trainer
    trainer = WeLTTrainer(
        model=model,
        args=eval_args,
        data_collator=collator,
        processing_class=processor,
        eval_metrics=eval_args.eval_metrics,
        generation_config=bytes_generation_config,
    )

    # Run evaluation on each split
    splits = [
        ("train", train_dataset, train_pred_file_path),
        ("eval", eval_dataset, eval_pred_file_path),
        ("test", test_dataset, test_pred_file_path),
    ]

    all_results = {}
    for split_name, dataset, pred_file_path in splits:
        logger.info(f"Evaluating {split_name} split...")

        # Get dataloader for this split
        trainer.eval_dataset = dataset
        dataloader = trainer.get_eval_dataloader()

        # Run evaluation loop
        output = trainer.evaluation_loop(
            dataloader,
            description=f"{split_name.capitalize()} evaluation",
            metric_key_prefix=split_name,
        )

        # Log metrics
        logger.info(f"{split_name.capitalize()} metrics: {output.metrics}")
        all_results[split_name] = output.metrics

        # Write predictions
        write_predictions(
            predictions=output.predictions,
            completions=output.label_ids,
            file_path=pred_file_path,
            split_name=split_name.capitalize(),
        )

        # Log example predictions
        if eval_args.log_examples_every and len(output.predictions) > 0:
            for i in range(min(eval_args.log_examples_every, len(output.predictions))):
                logger.info(
                    f"[{split_name}] Reference: {output.label_ids[i]}\tPrediction: {output.predictions[i]}"
                )

    # Save results
    for split_name in ["train", "eval", "test"]:
        results_file = model_dir / f"{split_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results[split_name], f, indent=4)
        logger.info(f"{split_name.capitalize()} results written to {results_file}")


if __name__ == "__main__":
    eval()
