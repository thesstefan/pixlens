from _typeshed import Incomplete
from transformers.modeling_utils import PreTrainedModel

class Blip2PreTrainedModel(PreTrainedModel):
    config_class = Incomplete
    base_model_prefix: str
    supports_gradient_checkpointing: bool

class Blip2ForConditionalGeneration(Blip2PreTrainedModel): ...
