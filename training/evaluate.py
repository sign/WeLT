
import json
import logging
import os
import sys
from pathlib import Path

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, HfArgumentParser
from transformers.utils import send_example_telemetry

from training.args_data import DataTrainingArguments
from training.args_eval import EvaluationArguments
from training.args_model import ModelArguments
from training.train import init_datasets, init_model, limit_dataset_size

logger = logging.getLogger(__name__)





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




def seperate_task(example, task_word: str):
    """
    Splits the input text into "text" and "label" based on the task_word.
        For example, if task_word is "<count>" and the input text is
        "<text>hello<count> H1 E1 L2 O1", then the output will be:
        {
            "text": "<text>hello<count>",
            "label": "H1 E1 L2 O1"
        }
        """
    sentence = example["text"]
    index = sentence.find(task_word)
    if index == -1:
        return {"text": sentence, "label": ""}
    cut = index + len(task_word) + 1
    return {"text": sentence[:cut], "label": sentence[cut:]}

def eval(args: list[str] | None | str = None): # noqa: C901

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = None

    # enable_optimizations()

    model_args, data_args, eval_args = parse_args_into_dataclasses(args)

    init_logging(eval_args)

    send_example_telemetry("run_clm", model_args, data_args)

    model, processor, collator = init_model(model_args, data_args, seed=42)

    model = model.to(device)


    text_datasets = init_datasets(data_args,
                                  cache_dir=cache_dir,
                                  trust_remote_code=model_args.trust_remote_code,
                                  do_train=False)


    # if "train" not in text_datasets:
    #     raise ValueError("--do_train requires a train dataset")  # noqa: TRY003
    # train_dataset = limit_dataset_size(text_datasets["train"],
    #                                     max_samples=data_args.max_train_samples,
    #                                     streaming=data_args.streaming)

    if "validation" not in text_datasets:
        raise ValueError("--do_eval requires a validation dataset")  # noqa: TRY003
    eval_dataset = limit_dataset_size(text_datasets["validation"],
                                        max_samples=eval_args.max_eval_samples_for_eval,
                                        streaming=data_args.streaming)

    if "test" not in text_datasets:
        raise ValueError("--do_eval requires a test dataset")  # noqa: TRY003
    test_dataset = limit_dataset_size(text_datasets["test"],
                                        max_samples=eval_args.max_test_samples_for_eval,
                                        streaming=data_args.streaming)


    # train_dataset = train_dataset.map(lambda x: seperate_task(x, eval_args.task_word))
    eval_dataset = eval_dataset.map(lambda x: seperate_task(x, eval_args.task_word))
    test_dataset = test_dataset.map(lambda x: seperate_task(x, eval_args.task_word))



    # Transform the datasets to the format expected by the model
    # if train_dataset:
    #     train_dataset = train_dataset.with_transform(processor)
    if eval_dataset:
        eval_dataset = eval_dataset.with_transform(processor)
    if test_dataset:
        test_dataset = test_dataset.with_transform(processor)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=eval_args.batch_size,
    #     shuffle=False,
    #     collate_fn=collator
    # )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        collate_fn=collator
    )


    bytes_generation_config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
    )


    ## PATHS
    model_dir = Path(model_args.model_name_or_path).parent \
        if model_args.model_name_or_path.endswith(".safetensors") \
            else Path(model_args.model_name_or_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # train_pred_file_path = model_dir / "train_predictions.jsonl"
    eval_pred_file_path = model_dir / "eval_predictions.jsonl"
    test_pred_file_path = model_dir / "test_predictions.jsonl"


    ## EVAL SET
    eval_pred_texts = []
    eval_actual_texts = []

    model.eval()
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(eval_loader, desc="Eval")):
            pred_text = model.generate(
                input_ids=batch["input_ids"].to(device),
                input_attention_mask = batch["input_attention_mask"].to(device),
                input_images = batch["input_images"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                input_images_dimensions = batch["input_images_dimensions"].to(device),
                processor=processor,
                bytes_generation_config=bytes_generation_config
            )
            eval_pred_texts.extend(pred_text)
            eval_actual_texts.extend(batch["label"])

            if eval_args.log_examples_every and i % eval_args.log_examples_every == 0:
                logger.info("Label: %s\tPred: %s\n", batch['label'][0], pred_text[0])


    # Write to a file
    with open(eval_pred_file_path, "w") as f:
        for ref, hyp in zip(eval_actual_texts, eval_pred_texts, strict=False):
            f.write(json.dumps({"reference": ref, "prediction": hyp}) + "\n")
    logger.info(f"Eval predictions written to {eval_pred_file_path}")


    ## TEST SET
    test_pred_texts = []
    test_actual_texts = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(test_loader, desc="Test")):
            pred_text = model.generate(
                input_ids=batch["input_ids"].to(device),
                input_attention_mask = batch["input_attention_mask"].to(device),
                input_images = batch["input_images"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                input_images_dimensions = batch["input_images_dimensions"].to(device),
                processor=processor,
                bytes_generation_config=bytes_generation_config
                )
            test_pred_texts.extend(pred_text)
            test_actual_texts.extend(batch["label"])

            if eval_args.log_examples_every and i % eval_args.log_examples_every == 0:
                logger.info("Label: %s\tPred: %s\n", batch['label'][0], pred_text[0])


    # Write to a file
    with open(test_pred_file_path, "w") as f:
        for ref, hyp in zip(test_actual_texts, test_pred_texts, strict=False):
            f.write(json.dumps({"reference": ref, "prediction": hyp}) + "\n")
    logger.info(f"Test predictions written to {test_pred_file_path}")


    eval_results = {}
    test_results = {}
    for metric_name in eval_args.eval_metrics:
        metric = evaluate.load(metric_name)
        is_sacrebleu = metric_name.lower() == "sacrebleu"

        eval_result = metric.compute(
            predictions=eval_pred_texts,
            references=[[r] for r in eval_actual_texts] if is_sacrebleu else eval_actual_texts
        )
        logger.info("Eval %s: %s", metric_name, eval_result)
        eval_results[metric_name] = eval_result

        test_result = metric.compute(
            predictions=test_pred_texts,
            references=[[r] for r in test_actual_texts] if is_sacrebleu else test_actual_texts
        )
        logger.info("Test %s: %s", metric_name, test_result)
        test_results[metric_name] = test_result

    # Save evals
    with open(model_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    logger.info(f"Eval results written to {model_dir / 'eval_results.json'}")
    with open(model_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    logger.info(f"Test results written to {model_dir / 'test_results.json'}")



if __name__ == "__main__":
    eval()
