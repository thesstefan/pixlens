import abc
import dataclasses
from typing import Protocol

import torch


@dataclasses.dataclass
class ImageDescription(Protocol):
    def caption(self) -> str:
        ...


class ImageDescriptorModel(Protocol):
    def image_caption(
        self,
        image_path: str,
    ) -> ImageDescription:
        ...
