# ruff: noqa: FBT001, PLR0913

from _typeshed import Incomplete
import torch

from transformers.processing_utils import ProcessorMixin

class Owlv2Processor(ProcessorMixin):
    def __call__(
        self,
        text: Incomplete | None = ...,
        images: Incomplete | None = ...,
        query_images: Incomplete | None = ...,
        padding: str = ...,
        return_tensors: str = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
    def post_process_object_detection(
        self,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> list[dict[str, torch.Tensor]]: ...
