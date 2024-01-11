import enum
from abc import abstractmethod

from PIL import Image

from pixlens.base_model import BaseModel
from pixlens.evaluation.interfaces import Edit


class ImageEditingPromptType(enum.Enum):
    INSTRUCTION = 1
    DESCRIPTION = 2


class PromptableImageEditingModel(BaseModel):
    @property
    @abstractmethod
    def prompt_type(self) -> ImageEditingPromptType:
        ...

    @abstractmethod
    def generate_prompt(self, edit: Edit) -> str:
        ...

    @abstractmethod
    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        ...
