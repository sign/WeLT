from pathlib import Path

import torch
from transformers import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from welt.model import WordLatentTransformerForCausalLM
from welt.processor import TextImageProcessor


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def sample(model_path: Path):
    last_checkpoint = get_last_checkpoint(model_path)
    model: WordLatentTransformerForCausalLM = \
        WordLatentTransformerForCausalLM.from_pretrained(last_checkpoint, attn_implementation="flash_attention_2")
    processor = TextImageProcessor.from_pretrained(model_path)

    model.eval()

    texts = [
        # Texts from validation set
        "<en>\x0EWhat's wrong?\x0F<he> ",
        "<en>\x0EYOu dOn't know the half Of it.\x0F<he> ",  # look! mixed case
        "<en>\x0E- No, just said that you were acting... aggressive.\x0F<he> ",
        "<en>\x0E-l'm a deputy.\x0F<he> ",  # look! "l" instead of "I", no space.
        "<en>\x0EWell, the good news is Joe wasn't cheating on you.\x0F<he> ",
        "<en>\x0E- Mm-hmm. No wonder women won't flirt with me.\x0F<he> ",
        "<en>\x0EYou understand you'll be working with your father.\x0F<he> ",
        "<en>\x0EYou wanted freedom, you got it.\x0F<he> ",
        # Not from validation set, just wanted to try a long named entity
        "<en>\x0EAlexander Hamilton\x0F<he> ",
    ]

    inputs = processor(texts, collated=True, packed=False)

    outputs = model.generate(
        **inputs,
        processor=processor,
        max_generated_words=32,
        bytes_generation_config=GenerationConfig(num_beams=2)  # Sample with beam search, for example
    )
    for text, output in zip(texts, outputs, strict=False):
        print(f"Generated for '{text}': {output}")


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "output" / "signed-to-spoken"
    sample(model_path)
