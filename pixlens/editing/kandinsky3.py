from PIL import Image
import torch
from torch._tensor import Tensor

from pixlens.editing import interfaces
from pixlens.editing.impl.kandinsky.kandinsky3 import get_inpainting_pipeline
from pixlens.editing.impl.kandinsky.kandinsky3.utils import prepare_mask
from pixlens.evaluation.interfaces import Edit
from pixlens.editing.utils import (
    generate_description_based_prompt,
    log_model_if_not_in_cache,
)


class Kandinsky3(interfaces.PromptableImageEditingModel):
    device: torch.device | None
    num_inference_steps: int
    seed: int

    def __init__(
        self,
        device: torch.device | None = None,
        num_inference_steps: int = 100,
        seed: int = 0,
    ) -> None:
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.seed = seed

    def load_kandinsky3(self) -> None:
        self.model = get_inpainting_pipeline(self.device, fp16=True)

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        del edit_info
        image = Image.open(image_path)
        mask = torch.zeros_like(image.size)
        images = self.model(
            prompt,
            image,
            mask,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            generator=torch.manual_seed(self.seed),
        )
        return images[0]

    def get_latent(self, prompt: str, image_path: str) -> Tensor:
        raise NotImplementedError

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.DESCRIPTION

    def generate_prompt(self, edit: Edit) -> str:
        return generate_description_based_prompt(edit)
