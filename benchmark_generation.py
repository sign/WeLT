"""Benchmark generation speed for WordLatentTransformerForCausalLM."""

import time
import torch
from datasets import Dataset

from tests.test_model import dataset_to_batch, make_dataset, setup_tiny_model


def benchmark_generation(num_runs: int = 10, warmup_runs: int = 2):
    """Benchmark generation speed."""
    print("Setting up model...")
    model, processor, collator = setup_tiny_model()
    model.eval()

    # Test batches of varying complexity
    test_cases = [
        (["a"], "single_short"),
        (["hello world"], "single_medium"),
        (["a", "b", "c", "d"], "batch_4_short"),
        (["hello", "world", "foo", "bar"], "batch_4_medium"),
        (["a", "hello world", "test"], "batch_3_mixed"),
    ]

    results = {}

    for texts, name in test_cases:
        print(f"\nBenchmarking: {name} ({texts})")
        dataset = make_dataset(texts)
        batch = dataset_to_batch(model, processor, collator, dataset)
        processor.max_word_length = 5

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model.generate(
                    input_ids=batch["input_ids"],
                    input_attention_mask=batch["input_attention_mask"],
                    input_images=batch["input_images"],
                    input_images_dimensions=batch["input_images_dimensions"],
                    attention_mask=batch["attention_mask"],
                    processor=processor,
                    max_generated_words=5
                )

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch["input_ids"],
                    input_attention_mask=batch["input_attention_mask"],
                    input_images=batch["input_images"],
                    input_images_dimensions=batch["input_images_dimensions"],
                    attention_mask=batch["attention_mask"],
                    processor=processor,
                    max_generated_words=5
                )
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        results[name] = {
            "avg_ms": avg_time * 1000,
            "min_ms": min_time * 1000,
            "max_ms": max_time * 1000,
            "batch_size": len(texts),
            "outputs": outputs
        }

        print(f"  Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")

    return results


def print_results_table(results: dict, version: str):
    """Print results as a markdown table."""
    print(f"\n## {version}\n")
    print("| Test Case | Batch Size | Avg (ms) | Min (ms) | Max (ms) |")
    print("|-----------|------------|----------|----------|----------|")
    for name, data in results.items():
        print(f"| {name} | {data['batch_size']} | {data['avg_ms']:.2f} | {data['min_ms']:.2f} | {data['max_ms']:.2f} |")


if __name__ == "__main__":
    results = benchmark_generation(num_runs=20, warmup_runs=5)
    print_results_table(results, "Current Version")
