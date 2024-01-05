import enum
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image

from pixlens.base_model import BaseModel
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


class ImageEditingPromptType(enum.Enum):
    INSTRUCTION = 1
    DESCRIPTION = 2


class PromptableImageEditingModel(ABC, BaseModel):
    @property
    @abstractmethod
    def prompt_type(self) -> ImageEditingPromptType:
        ...

    @abstractmethod
    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        ...

    @abstractmethod
    def generate_prompt(self, edit: Edit) -> str:
        ...

    # FIXME(thesstefan): Checking if the image exists and/or caching
    #                    should not be done at this level.
    def check_if_image_exists(
        self,
        prompt: str,
        image_path: str,
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

    def edit(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        image_exists_bool, path_of_image = self.check_if_image_exists(
            prompt,
            image_path,
        )
        if image_exists_bool:
            logging.info("Image already exists, loading...")
            edited_image = Image.open(path_of_image)
        else:
            logging.info("Editing image...")
            edited_image = self.edit_image(prompt, image_path, edit_info)
            path_of_image.parent.mkdir(parents=True, exist_ok=True)
            edited_image.save(path_of_image)
            logging.info("Image (and annotation) saved to %s", path_of_image)
        return edited_image

    @abstractmethod
    def get_latent(self, prompt: str, image_path: str) -> torch.Tensor:
        ...
