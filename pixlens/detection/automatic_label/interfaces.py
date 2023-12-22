import dataclasses
from typing import Protocol


@dataclasses.dataclass
class ImageCaption:
    caption: str


class ImageDescriptorModel(Protocol):
    def image_caption(
        self,
        image_path: str,
    ) -> ImageCaption:
        ...
