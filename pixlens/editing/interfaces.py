import dataclasses
import logging
from abc import abstractmethod
from typing import Protocol
from pathlib import Path

from PIL import Image
from PIL.Image import Image as PILImage

from pixlens.utils import utils


@dataclasses.dataclass
class ImageEditingOutput:
    output_image: PILImage
    prompt: str


class PromptableImageEditingModel(Protocol):
    @abstractmethod
    def get_model_name(self) -> str:
        ...

    @abstractmethod
    def edit_image(self, prompt: str, image_path: str) -> ImageEditingOutput:
        ...

    def edit(self, prompt: str, image_path: str) -> ImageEditingOutput:
        cache_dir = utils.get_cache_dir()
        model_dir = self.get_model_name().replace("/", "--")
        model_dir = "models--" + model_dir
        full_path = cache_dir / model_dir / Path(image_path).stem / prompt
        full_path = full_path.with_suffix(".png")
        if full_path.is_file():
            logging.info("Using cached edited image")
            return ImageEditingOutput(
                output_image=Image.open(full_path),
                prompt=prompt,
            )

        logging.info("Editing image...")
        edited_image = self.edit_image(prompt, image_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        edited_image.output_image.save(full_path)
        return edited_image
