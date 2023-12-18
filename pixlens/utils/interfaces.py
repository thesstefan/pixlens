import dataclasses
from typing import Protocol
from PIL import Image
import torch


@dataclasses.dataclass
class DetectionOutput:
    bounding_boxes: torch.Tensor
    logits: torch.Tensor
    phrases: list[str]


@dataclasses.dataclass
class SegmentationOutput:
    masks: torch.Tensor
    logits: torch.Tensor


@dataclasses.dataclass
class ImageEditingOutput:
    image: Image.Image
    prompt: str
    # we could maybe add more stuff?


class PromptableDetectionModel(Protocol):
    def detect(self, prompt: str, image_path: str) -> DetectionOutput:
        ...


class BBoxSegmentationModel(Protocol):
    def segment(self, bbox: torch.Tensor, image_path: str) -> SegmentationOutput:
        ...


class PromptableSegmentationModel(Protocol):
    def segment(self, prompt: str, image_path: str) -> SegmentationOutput:
        ...


class PromptableImageEditingModel(Protocol):
    def edit(self, prompt: str, image_path: str) -> ImageEditingOutput:
        ...
