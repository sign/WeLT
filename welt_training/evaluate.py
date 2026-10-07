"""
Generation evaluation of an exported WeLT model, on the validation split of a training config.
The config's `dataset_text_template` must be [prefix, completion]: generate from the prefix, compare to the completion.

    python -m welt_training.evaluate <export_dir> [--max_samples 256] [--output results.json]
"""
import argparse
import json
import os
import time

from sacrebleu.metrics import CHRF

from welt.inference import WeLTGenerator
from welt_training.data_utils import TextDataConfig, load_raw_datasets
from welt_training.extendable_yaml import CONFIG_FILE_NAME, load_yaml


def evaluate(export_dir: str, max_samples: int = 256, max_generated_words: int = 64) -> dict:
    config = load_yaml(os.path.join(export_dir, CONFIG_FILE_NAME))
    data = {k: v for k, v in config["data"].items() if k in TextDataConfig.__dataclass_fields__}
    data = TextDataConfig(**{**data, "max_eval_samples": max_samples})
    prefix_template, completion_template = data.dataset_text_template
    validation = load_raw_datasets(data)["validation"]
    validation = validation.select(range(min(max_samples, len(validation))))
    prefixes = [prefix_template.format(**example) for example in validation]
    references = [completion_template.format(**example) for example in validation]

    generator = WeLTGenerator(export_dir)
    generator.generate(prefixes[:2], max_generated_words=2)  # Warmup
    start = time.perf_counter()
    predictions = generator.generate(prefixes, max_generated_words=max_generated_words)
    elapsed = time.perf_counter() - start

    num_words = sum(len(generator.processor.pretokenize(p)) - 1 for p in predictions)
    return {
        "samples": len(references),
        "exact_match": sum(p.strip() == r.strip()
                           for p, r in zip(predictions, references, strict=True)) / len(references),
        "chrf": CHRF().corpus_score(predictions, [references]).score,
        "generation_seconds": elapsed,
        "generated_words_per_second": num_words / elapsed,
        "examples": [{"prefix": p, "reference": r, "prediction": o}
                     for p, r, o in list(zip(prefixes, references, predictions, strict=True))[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--max_generated_words", type=int, default=64)
    parser.add_argument("--output", help="Write the results as JSON")
    args = parser.parse_args()

    results = evaluate(args.export_dir, args.max_samples, args.max_generated_words)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
