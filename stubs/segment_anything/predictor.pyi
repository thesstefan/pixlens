# ruff: noqa: FBT001, PLR0913

import numpy.typing as npt
import torch
from _typeshed import Incomplete
from segment_anything.modeling import Sam as Sam

class SamPredictor:
    model: Incomplete
    transform: Incomplete
    def __init__(self, sam_model: Sam) -> None: ...
    def set_image(
        self,
        image: npt.NDArray[Incomplete],
        image_format: str = ...,
    ) -> None: ...
    original_size: Incomplete
    input_size: Incomplete
    features: Incomplete
    is_image_set: bool
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: tuple[int, ...],
    ) -> None: ...
    def predict(
        self,
        point_coords: npt.NDArray[Incomplete] | None = ...,
        point_labels: npt.NDArray[Incomplete] | None = ...,
        box: npt.NDArray[Incomplete] | None = ...,
        mask_input: npt.NDArray[Incomplete] | None = ...,
        multimask_output: bool = ...,
        return_logits: bool = ...,
    ) -> tuple[
        npt.NDArray[Incomplete],
        npt.NDArray[Incomplete],
        npt.NDArray[Incomplete],
    ]: ...
    def predict_torch(
        self,
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        boxes: torch.Tensor | None = ...,
        mask_input: torch.Tensor | None = ...,
        multimask_output: bool = ...,
        return_logits: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def get_image_embedding(self) -> torch.Tensor: ...
    @property
    def device(self) -> torch.device: ...
    orig_h: Incomplete
    orig_w: Incomplete
    input_h: Incomplete
    input_w: Incomplete
    def reset_image(self) -> None: ...
