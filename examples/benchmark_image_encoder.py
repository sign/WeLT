import torch
from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS
from pixel_renderer import PixelRendererProcessor
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from words_segmentation.tokenizer import WordsSegmentationTokenizer

from welt.processor import TextImageProcessor
from welt.vision.batch_image_encoder import encode_images, encode_padded_images
from welt.vision.masked_vit_patcher import maybe_patch_vit_model
from welt.vision.navit import NaViTConfig, NaViTModel

text = """
Some weights of ViTModel were not initialized from the model checkpoint at WinKawaks/vit-tiny-patch16-224 and are
newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[INFO|trainer.py:2523] 2025-09-08 20:09:21,135 >> ***** Running training *****
[INFO|trainer.py:2524] 2025-09-08 20:09:21,138 >>   Num examples = 138,167
[INFO|trainer.py:2525] 2025-09-08 20:09:21,138 >>   Num Epochs = 39
[INFO|trainer.py:2526] 2025-09-08 20:09:21,138 >>   Instantaneous batch size per device = 176
[INFO|trainer.py:2529] 2025-09-08 20:09:21,138 >>   Total train batch size (w. parallel,
distributed & accumulation) = 176
[INFO|trainer.py:2530] 2025-09-08 20:09:21,138 >>   Gradient Accumulation steps = 1
[INFO|trainer.py:2531] 2025-09-08 20:09:21,138 >>   Total optimization steps = 30,000
[INFO|trainer.py:2532] 2025-09-08 20:09:21,139 >>   Number of trainable parameters = 30,286,208"""

words = text.strip().split()  # 117~ words
print(len(words))

font_config = FontConfig(sources=FONTS_NOTO_SANS)
processor = TextImageProcessor(
    pretokenizer=WordsSegmentationTokenizer(),
    tokenizer=UTF8Tokenizer(),
    renderer=PixelRendererProcessor(font=font_config),
    image_processor=AutoImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224", use_fast=True),
)

renders, dimensions = processor.render_texts(words)

vit_model = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")

patched_vit_model = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
maybe_patch_vit_model(patched_vit_model)

# Similar model~
navit_model = NaViTModel(NaViTConfig(
    image_size=512,  # Max size for pos embeddings
    patch_size=vit_model.config.patch_size,
    hidden_size=vit_model.config.hidden_size,
    dim=vit_model.config.intermediate_size,
    depth=vit_model.config.num_hidden_layers // 2,
    heads=vit_model.config.num_attention_heads,
    mlp_dim=128,
))

num_steps = 10
batch_size = 1
batch = torch.stack([renders] * batch_size)
dimensions_batch = torch.stack([dimensions] * batch_size)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# if torch.backends.mps.is_available():
#     device = "mps"

vit_model = vit_model.to(device)
patched_vit_model = patched_vit_model.to(device)
navit_model = navit_model.to(device)
batch = batch.to(device)
dimensions_batch = dimensions_batch.to(device)

for _ in tqdm(range(num_steps), desc="Patched ViT model with padding"):
    encode_padded_images(image_encoder=patched_vit_model, input_images=batch)

for _ in tqdm(range(num_steps), desc="ViT Model grouped"):
    encode_images(image_encoder=vit_model, input_images=batch, input_images_dimensions=dimensions_batch)

for _ in tqdm(range(num_steps), desc="ViT Model with padding"):
    encode_padded_images(image_encoder=vit_model, input_images=batch)

for _ in tqdm(range(num_steps), desc="NaViT Model grouped"):
    encode_images(image_encoder=navit_model, input_images=batch, input_images_dimensions=dimensions_batch)
