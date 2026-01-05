import os
from pathlib import Path

import torch
from transformers.trainer_utils import get_last_checkpoint

from tests.test_model import setup_tiny_model
from welt.model import WordLatentTransformerForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def sample(model_name_or_path: str|Path):
    print("Loading model from:", model_path)
    if os.path.exists(str(model_path)):
        model_name_or_path = get_last_checkpoint(model_path)
        print("Checkpoint found at:", model_name_or_path)
    model: WordLatentTransformerForCausalLM = \
        WordLatentTransformerForCausalLM.from_pretrained(model_name_or_path,
                                                         dtype=torch.bfloat16,
                                                         device_map=DEVICE)
    model.enable_optimizations()

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
    inputs = {k: v.to(device=DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        processor=processor,
        max_generated_words=32,
    )
    for text, output in zip(texts, outputs, strict=False):
        print(f"Generated for '{text}': {output}")


if __name__ == "__main__":
    # model_path = Path(__file__).parent.parent / "output" / "string-repetition-tiny"
    model_path = "sign/WeLT-string-repetition"
    sample(model_path)

# python -m welt_training.sample --model_name_or_path sign/WeLT-string-repetition
