from typing import Optional, TypeVar
from _typeshed import Incomplete
from transformers.modeling_utils import PreTrainedModel

_Self = TypeVar("_Self", bound="BlipPreTrainedModel")

class BlipPreTrainedModel(PreTrainedModel):
    config_class = Incomplete
    base_model_prefix: str
    supports_gradient_checkpointing: bool
    def _reorder_cache(
        self, past_key_values: Incomplete, beam_idx: Incomplete
    ) -> Incomplete: ...
    def forward(
        self, *args: Incomplete, **kwargs: Incomplete
    ) -> Incomplete: ...
    def get_position_embeddings(
        self, *args: Incomplete, **kwargs: Incomplete
    ) -> Incomplete: ...
    def prepare_inputs_for_generation(
        self: _Self,
        input_ids: Incomplete,
        past: Optional[Incomplete] = None,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def resize_position_embeddings(
        self,
        new_num_position_embeddings: int,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> None: ...

class BlipForConditionalGeneration(BlipPreTrainedModel):
    def __init__(self, config: Incomplete) -> None: ...
