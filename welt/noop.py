"""
This model creates NoOp classes for frequently used Huggingface Transformers components,
to allow for specifying we do not want to use a component.
"""

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    BaseImageProcessor,
    PretrainedConfig,
    PreTrainedModel,
)


class NoopImageProcessor(BaseImageProcessor):
    name = "noop-image-processor"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, **unused_kwargs):
        raise NotImplementedError()


class NoopConfig(PretrainedConfig):
    model_type = "noop_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = 0


class NoopModel(PreTrainedModel):
    config_class = NoopConfig

    def __init__(self, config: NoopConfig):
        super().__init__(config=config)


AutoImageProcessor.register(NoopConfig, NoopImageProcessor)
AutoConfig.register(NoopConfig.model_type, NoopConfig)
AutoModel.register(NoopConfig, NoopModel)
