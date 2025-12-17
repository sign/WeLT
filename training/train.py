# Heavily adapted from
# https://github.com/huggingface/transformers/edit/main/examples/pytorch/language-modeling/run_clm.py
import logging
import math
import os
import sys

import datasets
import torch
import transformers
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from safetensors.torch import load_model
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import pack_dataset

from training.args_data import DataTrainingArguments
from training.args_model import ModelArguments
from training.args_trainer import WeLTTrainingArguments
from training.extendable_yaml import resolve_yaml_file
from training.freeze_callback import FreezeWarmupCallback
from training.trainer import WeLTTrainer
from welt.model_utils import setup_model

logger = logging.getLogger(__name__)


def enable_optimizations():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    # For debugging purposes only:
    # torch.autograd.set_detect_anomaly(True)

    # TODO --use_cuda_graphs true ?

    # TODO pin_memory_device="cuda" (PyTorch ≥2.3)

    # TODO
    #     torch_compile=True,
    #     torch_compile_backend="inductor",
    #     torch_compile_mode="default",

    # TODO use accelerate launch


def split_streaming_dataset(
        full_streaming_dataset,
        validation_percentage: int = 5,
) -> IterableDatasetDict:
    """
    Splits a streaming dataset into
    training and validation IterableDatasets, and supports methods like .map(), .filter(),
    .take() and properties like .features on the resulting streams.

    Args:
        full_streaming_dataset (Dataset): The name of the dataset to load (e.g., "HuggingFaceFW/fineweb").
        validation_percentage (int): The proportion of the dataset to be used for validation split.

    Returns:
        IterableDatasetDict: An IterableDatasetDict containing
            two IterableDataset objects: (train_stream, validation_stream).
    """
    if not (0 < validation_percentage < 100):
        raise ValueError(  # noqa: TRY003
            f"validation_percentage must be between 0 and 100 (exclusive). Passed: {validation_percentage}"
        )

    def split_generator(is_train: bool):
        for i, example in enumerate(full_streaming_dataset):
            if is_train:
                if i % 100 > validation_percentage:
                    yield example
            else:
                if i % 100 < validation_percentage:
                    yield example

    features = full_streaming_dataset.features
    train_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": True}, features=features)
    validation_stream = IterableDataset.from_generator(
        split_generator, gen_kwargs={"is_train": False}, features=features
    )

    return IterableDatasetDict({"train": train_stream, "validation": validation_stream})


def parse_args_into_dataclasses(args: list[str] | None | str = None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, WeLTTrainingArguments))
    # If we pass only one argument to the script and it's the path to a json or yaml file,
    # let's parse it to get our arguments.
    if isinstance(args, str):
        resolved_path = resolve_yaml_file(os.path.abspath(args))
        return parser.parse_yaml_file(yaml_file=resolved_path)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        resolved_path = resolve_yaml_file(os.path.abspath(sys.argv[1]))
        return parser.parse_yaml_file(yaml_file=resolved_path)
    else:
        return parser.parse_args_into_dataclasses(args=args)


def init_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, " +
        f"device: {training_args.device}, " +
        f"n_gpu: {training_args.n_gpu}, " +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, " +
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def init_model(model_args: ModelArguments, data_args: DataTrainingArguments, seed: int):
    # Set seed before initializing model.
    set_seed(seed)

    # Initialize the model
    model, processor, collator = setup_model(
        image_encoder_name=model_args.image_encoder_model_name_or_path,
        bytes_encoder_name=model_args.bytes_encoder_model_name_or_path,
        latent_transformer_name=model_args.latent_transformer_model_name_or_path,
        bytes_decoder_name=model_args.bytes_decoder_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        dtype=model_args.dtype,
        seed=seed,
        load_pretrained=model_args.load_pretrained,
        max_word_length=data_args.max_word_length,
        pretokenizer_name=model_args.pretokenizer_name,
    )

    # Load the model from a local path if provided
    if model_args.model_name_or_path:
        load_model(model, model_args.model_name_or_path)

    return model, processor, collator


def detect_last_checkpoint(training_args: TrainingArguments):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    return last_checkpoint


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
            single_text_template = template \
                if is_single_text_template else "".join(template)

            def mapping_fn(example):
                if is_single_text_template or split_name == "train":
                    return {"text": single_text_template.format(**example)}

                prefix = template[0].format(**example)
                completion = template[1].format(**example)
                return {
                    "text": f"{prefix}{completion}",  # Full text for training loss calculation
                    "prefix": prefix,  # For generation
                    "completion": completion,  # Reference for metrics
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


def limit_dataset_size(dataset, max_samples: int | None = None, streaming: bool = False):
    if max_samples is not None:
        if streaming:
            dataset = dataset.take(max_samples)
        elif max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

    return dataset




def train(args: list[str] | None | str = None):  # noqa: C901
    cache_dir = None  # Use the default cache directory / Environment variable

    enable_optimizations()

    model_args, data_args, training_args = parse_args_into_dataclasses(args)

    init_logging(training_args)

    # Detecting last checkpoint.
    last_checkpoint = detect_last_checkpoint(training_args)

    # Initialize the model
    model, processor, collator = init_model(model_args, data_args, seed=training_args.seed)

    if data_args.max_sequence_length is not None:
        processor.max_seq_length = data_args.max_sequence_length

    # Save the processor to the output directory
    processor.save_pretrained(save_directory=training_args.output_dir, push_to_hub=False)

    # Load the datasets
    text_datasets = init_datasets(data_args,
                                  cache_dir=cache_dir,
                                  trust_remote_code=model_args.trust_remote_code,
                                  do_train=training_args.do_train)

    train_dataset = None
    if training_args.do_train:
        if "train" not in text_datasets:
            raise ValueError("--do_train requires a train dataset")  # noqa: TRY003
        train_dataset = limit_dataset_size(text_datasets["train"],
                                           max_samples=data_args.max_train_samples,
                                           streaming=data_args.streaming)

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in text_datasets:
            raise ValueError("--do_eval requires a validation dataset")  # noqa: TRY003
        eval_dataset = limit_dataset_size(text_datasets["validation"],
                                          max_samples=data_args.max_eval_samples,
                                          streaming=data_args.streaming)

    # Sequence packing
    if train_dataset:
        block_size = min(data_args.block_size or math.inf, processor.max_seq_length)
        train_dataset = processor.pretokenize_dataset(train_dataset)
        train_dataset = pack_dataset(train_dataset, seq_length=block_size)

    # Transform the datasets to the format expected by the model
    if train_dataset:
        train_dataset = train_dataset.with_transform(processor)
    if eval_dataset:
        eval_dataset = eval_dataset.with_transform(processor)

    # Initialize our Trainer
    # Note: WeLTTrainer computes accuracy and generation-based metrics internally
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        # Generation-based evaluation settings from training args
        eval_metrics=training_args.eval_metrics if training_args.do_eval else None,
        max_generated_words=training_args.generation_max_length or 50,
        log_samples=training_args.log_samples,
    )

    # Freeze the pretrained models for some steps
    trainer.add_callback(FreezeWarmupCallback(steps=model_args.warmup_freeze_steps, model=model))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        if data_args.streaming:
            metrics["train_samples"] = max_train_samples
        else:
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        if data_args.streaming:
            metrics["eval_samples"] = max_eval_samples
        else:
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    train()
