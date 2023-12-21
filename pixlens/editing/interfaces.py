import dataclasses
from typing import Protocol
from PIL import Image


@dataclasses.dataclass
class ImageEditingOutput:
    image: Image.Image
    prompt: str


class PromptableImageEditingModel(Protocol):
    def edit(self, prompt: str, image_path: str) -> ImageEditingOutput:
        ...
