"""Exercise the baseline CLM script with a local model and byte tokenizer."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


@pytest.mark.parametrize("streaming", [False, True])
def test_clm_reports_exact_bpb(tmp_path, monkeypatch, streaming):
    script = Path(__file__).parents[1] / "welt_training/experiments/machine-translation/run_clm.py"
    spec = importlib.util.spec_from_file_location("run_clm", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_clm", module)
    spec.loader.exec_module(module)

    texts = ["hello  world!", "café 🙂", " a ."]
    backend = ByteLevelBPETokenizer()
    backend.train_from_iterator(texts, vocab_size=257, special_tokens=["<|endoftext|>"])
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    backend.save(str(model_dir / "tokenizer.json"))
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"), model_max_length=8)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(42)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=8, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id))
    model.eval()
    model.save_pretrained(model_dir)
    data_path = tmp_path / "data.json"
    data_path.write_text("\n".join(json.dumps({"text": text}) for text in texts))
    output = tmp_path / "output"
    config_path = tmp_path / "eval.json"
    config_path.write_text(json.dumps({
        "model_name_or_path": str(model_dir), "validation_file": str(data_path),
        "output_dir": str(output), "do_eval": True, "use_cpu": True,
        "report_to": "none", "block_size": 8, "per_device_eval_batch_size": 2,
        "streaming": streaming, "eval_preserve_document_boundaries": True,
    }))
    monkeypatch.setattr(sys, "argv", [str(script), str(config_path)])
    module.main()
    metrics = json.loads((output / "eval_results.json").read_text())

    total_nats = 0.0
    total_bytes = 0
    chunks = 0
    for text in texts:
        ids = tokenizer.encode(text)
        for start in range(0, len(ids), 8):
            chunk = ids[start:start + 8]
            chunks += 1
            total_bytes += len(tokenizer.decode(
                chunk[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False).encode("utf-8"))
            if len(chunk) > 1:
                with torch.no_grad():
                    logits = model(torch.tensor([chunk])).logits[0, :-1]
                total_nats += torch.nn.functional.cross_entropy(
                    logits, torch.tensor(chunk[1:]), reduction="sum").item()
    assert metrics["eval_bits_per_byte"] == pytest.approx(total_nats / (total_bytes * math.log(2)), rel=1e-6)
    if not streaming:
        assert metrics["eval_samples"] == chunks
