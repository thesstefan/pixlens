from _typeshed import Incomplete
from transformers.modeling_utils import PreTrainedModel

class OwlViTPreTrainedModel(PreTrainedModel):
    config_class = Incomplete
    base_model_prefix: str
    supports_gradient_checkpointing: bool

class OwlViTForObjectDetection(OwlViTPreTrainedModel): ...
