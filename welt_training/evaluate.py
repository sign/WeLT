"""
Generation evaluation of a WeLT model served by welt-serve, on the validation split of its training config.
The config's `dataset_text_template` must be [prefix, completion]: generate from the prefix, compare to the completion.

    welt-evaluate <export_dir>/welt.yaml [--url http://localhost:8080] [--max_samples 256] [--output results.json]
"""
import argparse
import json
import time

from sacrebleu.metrics import CHRF

from welt.server import generate
from welt_training.data_utils import TextDataConfig
from welt_training.extendable_yaml import load_yaml


def evaluate(config_path: str, url: str, max_samples: int = 256, max_generated_words: int = 64) -> dict:
    config = load_yaml(config_path)
    data = TextDataConfig(**{k: v for k, v in config["data"].items() if k in TextDataConfig.__dataclass_fields__})
    if not isinstance(data.dataset_text_template, list) or len(data.dataset_text_template) != 2:
        raise ValueError("Generation evaluation needs data.dataset_text_template as [prefix, completion]")
    prefix_template, completion_template = data.dataset_text_template
    # Within the validation texts (without a validation split, held out from training)
    validation = list(data.examples("validation").take(max_samples))
    prefixes = [prefix_template.format(**example) for example in validation]
    references = [completion_template.format(**example) for example in validation]

    generate(url, prefixes, max_generated_words=2)  # Warmup, with the batch size of the measurement
    start = time.perf_counter()
    result = generate(url, prefixes, max_generated_words=max_generated_words)
    elapsed = time.perf_counter() - start
    predictions = result["outputs"]
    return {
        "config": config_path,
        "url": url,
        "max_generated_words": max_generated_words,
        "samples": len(references),
        "exact_match": sum(p.strip() == r.strip()
                           for p, r in zip(predictions, references, strict=True)) / len(references),
        "chrf": CHRF().corpus_score(predictions, [references]).score,
        "generation_seconds": elapsed,
        "generated_words_per_second": result["generated_words"] / elapsed,
        "examples": [{"prefix": p, "reference": r, "prediction": o}
                     for p, r, o in zip(prefixes, references, predictions, strict=True)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="The model's training config (welt.yaml in its export directory)")
    parser.add_argument("--url", default="http://localhost:8080", help="A running welt-serve")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--max_generated_words", type=int, default=64)
    parser.add_argument("--output", help="Write the results as JSON")
    args = parser.parse_args()

    results = evaluate(args.config, args.url, args.max_samples, args.max_generated_words)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        results.pop("examples")  # Only the summary on stdout
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
