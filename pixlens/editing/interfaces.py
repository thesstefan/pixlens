import dataclasses
from typing import Protocol

from PIL.Image import Image as PILImage


@dataclasses.dataclass
class ImageEditingOutput:
    input_image: PILImage
    output_image: PILImage
    prompt: str


class PromptableImageEditingModel(Protocol):
    def edit(self, prompt: str, image_path: str) -> ImageEditingOutput:
        ...
