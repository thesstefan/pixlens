import enum

import cv2  # type: ignore[import]
import numpy as np
import torch
from PIL import Image

from pixlens.editing import interfaces
from pixlens.editing.utils import (
    generate_instruction_based_prompt,
    log_model_if_not_in_cache,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


class (interfaces.PromptableImageEditingModel):
    model: StableDiffusionControlNetPipeline
    controlnet_type: ControlNetType
    device: torch.device | None
    num_inference_steps: int
    image_guidance_scale: float
    text_guidance_scale: float

    def __init__(  # noqa: PLR0913
        self,
        controlnet_type: ControlNetType = ControlNetType.CANNY,
        device: torch.device | None = None,
        num_inference_steps: int = 100,
        image_guidance_scale: float = 1.0,
        text_guidance_scale: float = 7.0,
        seed: int = 0,
        latent_guidance_scale: float = 25,
    ) -> None:
        self.model = load_controlnet(controlnet_type, device)
        self.device = device
        self.controlnet_type = controlnet_type
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale
        self.latent_guidance_scale = latent_guidance_scale
        self.seed = seed

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "controlnet_type": str(self.controlnet_type),
            "num_inference_steps": self.num_inference_steps,
            "image_guidance_scale": self.image_guidance_scale,
            "text_guidance_scale": self.text_guidance_scale,
            "latent_guidance_scale": self.latent_guidance_scale,
            "seed": self.seed,
        }

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.INSTRUCTION

    def generate_prompt(self, edit: Edit) -> str:
        return generate_instruction_based_prompt(edit)
