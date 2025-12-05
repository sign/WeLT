import torch
from transformers import AutoConfig, AutoModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from vit_pytorch.na_vit import NaViT


class NaViTConfig(PretrainedConfig):
    model_type = "navit"

    def __init__(
            self,
            image_size: int = 256,  # only used to set a default; NaViT handles var-size
            patch_size: int = 16,
            hidden_size: int = 512,
            dim: int = 512,
            depth: int = 6,
            heads: int = 8,
            mlp_dim: int = 1024,
            dropout: float = 0.0,
            emb_dropout: float = 0.0,
            token_dropout_prob: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.token_dropout_prob = token_dropout_prob


class NaViTModel(PreTrainedModel):
    config_class = NaViTConfig

    def __init__(self, config: NaViTConfig):
        super().__init__(config)

        self.navit = NaViT(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.hidden_size,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            token_dropout_prob=config.token_dropout_prob,
        )

        # Initialize weights via HF utility (won't disturb vit-pytorch-initialized weights unless uninitialized)
        self.post_init()

    def forward(self, images: list[torch.Tensor], **kwargs):
        # vit-pytorch NaViT returns shape (B, num_classes)
        logits = self.navit(images)

        return BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=logits
        )

AutoConfig.register(NaViTConfig.model_type, NaViTConfig)
AutoModel.register(NaViTConfig, NaViTModel)
