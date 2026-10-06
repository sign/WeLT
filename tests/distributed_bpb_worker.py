"""CPU worker for the end-to-end DDP evaluation regression test."""

import tempfile
from types import SimpleNamespace

import torch
from datasets import Dataset
from transformers import Seq2SeqTrainingArguments

from welt_training.trainer import WeLTTrainer


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(256))
        self.register_buffer("buffer", torch.zeros(1))
        self.config = SimpleNamespace(encoding="UTF-8", pad_token_id=0, keys_to_ignore_at_inference=[])

    @property
    def device(self):
        return self.weight.device

    def forward(self, labels_output):
        logits = self.weight.expand(*labels_output.shape, 256) + self.buffer
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, -2), labels_output.flatten(), ignore_index=0)
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, **kwargs):
        return ["prediction"] * len(kwargs["labels_output"])


class Processor:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=3)

    def __call__(self, batch, collated=False):
        if isinstance(batch, list):
            batch = {"text": batch}
        result = {"labels_output": [torch.tensor([[ord(text[0]), 3]]) for text in batch["text"]]}
        if collated:
            result["labels_output"] = torch.stack(result["labels_output"])
        else:
            for key in ["prefix", "completion"]:
                if key in batch:
                    result[key] = batch[key]
        return result


def collate(rows):
    result = {"labels_output": torch.stack([row["labels_output"] for row in rows])}
    for key in ["prefix", "completion"]:
        if key in rows[0]:
            result[key] = [row[key] for row in rows]
    return result


class CountMetric:
    def compute(self, predictions, references):
        assert len(predictions) == len(references) == 3
        return {"score": 1.0}


def main():
    with tempfile.TemporaryDirectory() as output_dir:
        for streaming in [False, True]:
            data = Dataset.from_dict({
                "text": ["a", "b", "c"], "prefix": ["a", "b", "c"], "completion": ["x", "y", "z"]})
            if streaming:
                data = data.to_iterable_dataset()
            args = Seq2SeqTrainingArguments(
                output_dir=output_dir, use_cpu=True, per_device_eval_batch_size=2,
                report_to="none", remove_unused_columns=False, predict_with_generate=True)
            trainer = WeLTTrainer(
                model=UniformModel(), processor=Processor(), data_collator=collate, args=args, log_samples=0)
            trainer.model = trainer.accelerator.prepare_model(trainer.model)
            trainer.model_wrapped = trainer.model
            assert isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
            trainer.loaded_metrics = {"count": CountMetric()}
            metrics = trainer.evaluate(data)
            assert metrics["eval_samples"] == 3
            assert abs(metrics["eval_bits_per_byte"] - 8.0) < 1e-6
            assert metrics["eval_count"] == 1.0
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
