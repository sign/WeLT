from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
)

from welt.config import Encoding


def listed_model(model_names):
    models_list = ", ".join(model_names.keys())
    return field(
        default=None,
        metadata={"help": "Any model implementing the architectures from the list: " + models_list},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    image_encoder_model_name_or_path: str | None = listed_model(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES)
    bytes_encoder_model_name_or_path: str | None = listed_model(MODEL_FOR_MASKED_LM_MAPPING_NAMES)
    latent_transformer_model_name_or_path: str | None = listed_model(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
    bytes_decoder_model_name_or_path: str | None = listed_model(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

    load_pretrained: bool = field(default=False, metadata={
        "help": "Whether to load the pretrained weights of the models specified in *_model_name_or_path."
    })

    warmup_freeze_steps: int = field(default=0, metadata={
        "help": "Steps to keep most modules frozen at start."
    })

    pretokenizer_name: str | None = field(default=None, metadata={
        "help": "Pretokenizer to use, defaults to https://github.com/sign/words-segmentation."
    })

    encoding: Encoding = field(default="UTF-8", metadata={
        "help": "Character encoding for the tokenizer. UTF-16/UTF-32 use multi-byte predictions."
    })

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )

    dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ModelArguments)
    parser.parse_args_into_dataclasses()
