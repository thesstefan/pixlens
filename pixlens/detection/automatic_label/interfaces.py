import abc
import dataclasses
from typing import Protocol

from PIL import Image
import torch


@dataclasses.dataclass
class ImageCaption:
    caption: str


class ImageDescriptorModel(Protocol):
    def image_caption(
        self,
        image: Image.Image,
    ) -> ImageCaption:
        ...


class CaptionIntoObjectsModel(Protocol):
    def extract_objects(self, caption: str) -> list[str]:
        ...


class ImageToObjects(abc.ABC, CaptionIntoObjectsModel):
    def __init__(
        self,
        image_descriptor_model: ImageDescriptorModel,
        caption_into_objects_model: CaptionIntoObjectsModel,
        device: torch.device | None = None,
    ) -> None:
        self.image_descriptor_model = image_descriptor_model
        self.caption_into_objects_model = caption_into_objects_model
        super().__init__()

    def image_to_objects(self, image: Image.Image) -> list[str]:
        caption = self.image_descriptor_model.image_caption(image)
        return self.caption_into_objects_model.extract_objects(caption.caption)
