from _typeshed import Incomplete
import torch

from transformers.processing_utils import ProcessorMixin

class Blip2Processor(ProcessorMixin):
    def __call__(
        self,
        text: Incomplete | None = ...,
        images: Incomplete | None = ...,
        query_images: Incomplete | None = ...,
        padding: str = ...,
        return_tensors: str = ...,
        **kwargs: Incomplete,
    ) -> Incomplete: ...
