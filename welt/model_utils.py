import importlib.util
from functools import partial

import torch
from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS
from pixel_renderer import PixelRendererProcessor
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    PretrainedConfig,
    enable_full_determinism,
    set_seed,
)
from utf8_tokenizer.tokenizer import UTF8Tokenizer, UTF16Tokenizer, UTF32Tokenizer
from words_segmentation.tokenizer import WordsSegmentationTokenizer

from welt.collator import collate_fn
from welt.config import Encoding, WordLatentTransformerConfig
from welt.model import WordLatentTransformerForCausalLM, logger
from welt.noop import NoopConfig, NoopImageProcessor
from welt.processor import TextImageProcessor
from welt.vision.navit import NaViTConfig


def print_model_summary(name: str, model):
    """Print a summary of the model's architecture."""
    if model is None:
        print(name, "is None")
        return
    total_params = sum(p.numel() for p in model.parameters())
    print(name, f"Total parameters: {total_params:,}")


def get_attn_implementation():
    if importlib.util.find_spec("flash_attn") is None:
        logger.warning("Flash Attention not available, using default attention")
        return None
    return "flash_attention_2"


CUSTOM_MODELS: dict[str, PretrainedConfig] = {
    "NaViT-tiny": NaViTConfig(
        patch_size=16,
        hidden_size=128,
        dim=64,
        depth=3,
        heads=4,
        mlp_dim=128,
        dropout=0.0,
        emb_dropout=0.0,
        token_dropout_prob=0.1,
    ),
    "NaViT-small": NaViTConfig(
        patch_size=16,
        hidden_size=512,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.0,
        emb_dropout=0.0,
        token_dropout_prob=0.1,
    )
}
CUSTOM_PROCESSORS_ALIAS: dict[str, str] = {
    "NaViT-tiny": "WinKawaks/vit-tiny-patch16-224",
    "NaViT-small": "WinKawaks/vit-tiny-patch16-224",
}


def get_model_config(model_name, config_path: str | None = None):
    """
    Get a model configuration.

    Args:
        model_name: Name of a pretrained model or custom model.
        config_path: Optional path to a JSON config file. Must include 'model_type'.
                    If provided, model_name is ignored.

    Returns:
        A PretrainedConfig instance.
    """
    if config_path is not None:
        import json
        with open(config_path) as f:
            config_dict = json.load(f)

        model_type = config_dict.pop("model_type", None)
        if model_type is None:
            raise ValueError(
                f"Config file {config_path} must include 'model_type' field "
                "(e.g., 'bert', 'gpt2', 'llama')"
            )
        # Create config from the model type and provided parameters
        return AutoConfig.for_model(model_type, **config_dict)

    if model_name is None:
        return NoopConfig()

    if model_name in CUSTOM_MODELS:
        return CUSTOM_MODELS[model_name]
    return AutoConfig.from_pretrained(model_name)


def setup_model(
        image_encoder_name="WinKawaks/vit-tiny-patch16-224",
        image_encoder_config: str | None = None,
        bytes_encoder_name="prajjwal1/bert-tiny",
        bytes_encoder_config: str | None = None,
        latent_transformer_name="EleutherAI/pythia-70m",
        latent_transformer_config: str | None = None,
        bytes_decoder_name="sign/utf8-lm-tiny",
        bytes_decoder_config: str | None = None,
        pretokenizer_name: str | None = None,
        encoding: Encoding = "UTF-8",
        trust_remote_code=False,
        modality_dropout=0.15,
        dtype=torch.float32,
        seed=42,
        load_pretrained=True,
        max_word_length=None,
        quiet=False,
):
    set_seed(seed, deterministic=True)
    enable_full_determinism(seed=seed, warn_only=True)

    # Load image processor - need a pretrained name even when using config
    if image_encoder_name is not None:
        image_processor_name = CUSTOM_PROCESSORS_ALIAS.get(image_encoder_name, image_encoder_name)
        image_processor = AutoImageProcessor.from_pretrained(image_processor_name, use_fast=True)
    elif image_encoder_config is not None:
        # When using config file, default to a standard ViT processor
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    else:
        image_processor = NoopImageProcessor()

    tokenizer_classes = {
        "UTF-8": UTF8Tokenizer,
        "UTF-16": UTF16Tokenizer,
        "UTF-32": UTF32Tokenizer,
    }
    tokenizer = tokenizer_classes[encoding]()

    config = WordLatentTransformerConfig(
        # All sub-configs are loaded from the respective model names or config files
        image_encoder=get_model_config(image_encoder_name, config_path=image_encoder_config),
        bytes_encoder=get_model_config(bytes_encoder_name, config_path=bytes_encoder_config),
        latent_transformer=get_model_config(latent_transformer_name, config_path=latent_transformer_config),
        bytes_decoder=get_model_config(bytes_decoder_name, config_path=bytes_decoder_config),
        # Other configuration parameters
        modality_dropout=modality_dropout,
        tokenizer_class=tokenizer.__class__.__name__,
        num_tokens=len(tokenizer),
        encoding=encoding,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )

    # Combine the models
    model = WordLatentTransformerForCausalLM(config,
                                             load_pretrained=load_pretrained,
                                             attn_implementation=get_attn_implementation())
    if not quiet:
        print_model_summary("Image Encoder", model.image_encoder)
        print_model_summary("Bytes Encoder", model.bytes_encoder)
        print_model_summary("Latent Transformer", model.latent_transformer)
        print_model_summary("Bytes Decoder", model.bytes_decoder)
        print_model_summary("Final Model", model)

    max_seq_length = getattr(model.latent_transformer.config, "max_position_embeddings", 1024)

    if max_word_length is None:
        decoder_max = getattr(model.bytes_decoder.config, "max_position_embeddings", 128)
        if model.bytes_encoder:
            encoder_max = getattr(model.bytes_encoder.config, "max_position_embeddings", float('inf'))
        else:
            encoder_max = float('inf')
        max_word_length = min(decoder_max, encoder_max)

    modified_max_length = max_word_length - 2  # Reserve space for BOS and EOS tokens
    if pretokenizer_name is not None:
        print(f"Using pretokenizer: {pretokenizer_name}")
        pretokenizer = AutoTokenizer.from_pretrained(pretokenizer_name,
                                                     use_fast=True,
                                                     trust_remote_code=trust_remote_code)
    else:
        if encoding == "UTF-32":
            print(f"Using pretokenizer: WordsSegmentationTokenizer(max_characters={modified_max_length})")
            pretokenizer = WordsSegmentationTokenizer(max_characters=modified_max_length)
        else:
            print(f"Using pretokenizer: WordsSegmentationTokenizer(max_bytes={modified_max_length})")
            pretokenizer = WordsSegmentationTokenizer(max_bytes=modified_max_length)

    font_config = FontConfig(sources=FONTS_NOTO_SANS)
    renderer = PixelRendererProcessor(font=font_config)

    processor = TextImageProcessor(
        pretokenizer=pretokenizer,
        tokenizer=tokenizer,
        renderer=renderer,
        image_processor=image_processor,
        max_seq_length=max_seq_length,
    )

    collator = partial(collate_fn, pad_value=tokenizer.pad_token_type_id)

    return model, processor, collator
