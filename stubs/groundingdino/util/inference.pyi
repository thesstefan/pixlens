# ruff: noqa: PLR0913

import numpy.typing as npt
import torch
from _typeshed import Incomplete

def preprocess_caption(caption: str) -> str: ...
def load_model(
    model_config_path: str,
    model_checkpoint_path: str,
    device: str = ...,
) -> torch.nn.Module: ...
def load_image(
    image_path: str,
) -> tuple[npt.NDArray[Incomplete], torch.Tensor]: ...
def predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str = ...,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]: ...
def annotate(
    image_source: npt.NDArray[Incomplete],
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: list[str],
) -> npt.NDArray[Incomplete]: ...
