from pathlib import Path

import torch
from transformers.trainer_utils import get_last_checkpoint

from tests.test_model import setup_tiny_model
from welt.model import WordLatentTransformerForCausalLM


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def sample(model_path: Path):
    print("Loading model from:", model_path)
    last_checkpoint = get_last_checkpoint(model_path)
    print("Checkpoint found at:", last_checkpoint)
    model: WordLatentTransformerForCausalLM = \
        WordLatentTransformerForCausalLM.from_pretrained(last_checkpoint)
    # TODO load processor from_pretrained
    # processor = TextImageProcessor.from_pretrained(model_path)
    _, processor, _ = setup_tiny_model(image_encoder_name=None)


    model.eval()

    texts = [
        # Texts from validation set
        (
            "<text>\x0EWouldn't it be more cruel for society to let people die... - "
            "... when with some effort it could save them?\x0F<repeat> "
        ),
        "<text>\x0ESuperman's exact opposite who lives in the backwards Bizarro World.\x0F<repeat> ",
        "<text>\x0EYOu dOn't know the half Of it.\x0F<repeat> ",
    ]

    inputs = processor(texts, collated=True, packed=False)

    outputs = model.generate(
        **inputs,
        processor=processor,
        max_generated_words=32,
        # bytes_generation_config=GenerationConfig(num_beams=2)  # Sample with beam search, for example
    )
    for text, output in zip(texts, outputs, strict=False):
        print(f"Generated for '{text}': {output}")


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "output" / "string-repetition-tiny"
    sample(model_path)
