import logging
from abc import abstractmethod, ABC
from typing import Protocol
from pathlib import Path

from PIL import Image

from pixlens.utils import utils


class Model(ABC):
    @abstractmethod
    def get_model_name(self) -> str:
        pass


class ImageEditor(Protocol):
    @abstractmethod
    def edit_image(self, prompt: str, image_path: str) -> Image.Image:
        pass


class PromptableImageEditingModel(Model, ImageEditor):
    @abstractmethod
    def edit_image(self, prompt: str, image_path: str) -> Image.Image:
        ...

    def check_if_image_exists(
        self, prompt: str, image_path: str
    ) -> tuple[bool, Path]:
        """Check if the image exists and return a tuple of (bool, path).

        The bool value is True if the image exists,
        and the path is where the image is stored.
        """
        cache_dir = utils.get_cache_dir()
        model_dir = self.get_model_name().replace("/", "--")
        model_dir = "models--" + model_dir
        full_path = cache_dir / model_dir / Path(image_path).stem / prompt
        full_path = full_path.with_suffix(".png")
        return full_path.exists(), full_path

    # TODO: fix this, check if image exists and saving should be done somewhere else
    def edit(self, prompt: str, image_path: str) -> Image.Image:
        image_exists_bool, path_of_image = self.check_if_image_exists(
            prompt, image_path
        )
        if image_exists_bool:
            logging.info("Image already exists, loading...")
            edited_image = Image.open(path_of_image)
        else:
            logging.info("Editing image...")
            edited_image = self.edit_image(prompt, image_path)
            path_of_image.parent.mkdir(parents=True, exist_ok=True)
            edited_image.save(path_of_image)
        return edited_image
