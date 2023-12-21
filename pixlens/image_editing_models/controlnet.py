import PIL
import logging
import requests
import enum
import torch
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
import cv2
from PIL import Image
import numpy as np

from pixlens.utils import utils
from pixlens.utils import interfaces
from pixlens.image_editing_models.utils import log_model_if_not_in_cache


class ControlNetType(enum.StrEnum):
    BASE = "lllyasviel/sd-controlnet-canny"


class StableDiffusionType(enum.StrEnum):
    BASE = "runwayml/stable-diffusion-v1-5"


def load_controlnet(
    model_type: ControlNetType, device: torch.device | None = None
) -> StableDiffusionControlNetPipeline:
    path_to_cache = utils.get_cache_dir()
    log_model_if_not_in_cache(model_type, path_to_cache)
    controlnet = ControlNetModel.from_pretrained(model_type, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        StableDiffusionType.BASE,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=path_to_cache,
    )

    pipe.to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    return pipe


class ControlNet(interfaces.PromptableImageEditingModel):
    device: torch.device | None

    def __init__(
        self,
        pix2pix_type: ControlNetType,
        device: torch.device | None = None,
    ) -> None:
        self.device = device
        self.model = load_controlnet(pix2pix_type, device)

    def prepare_image(self, image_path: str) -> PIL.Image.Image:
        image = PIL.Image.open(image_path)
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image

    def edit(
        self,
        prompt: str,
        image_path: str,
        *,
        num_inference_steps: int = 100,
        image_guidance_scale: float = 1.0,
    ) -> interfaces.ImageEditingOutput:
        input_image = self.prepare_image(image_path)
        output_image = self.model(
            prompt,
            input_image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
        ).images[0]
        return interfaces.ImageEditingOutput(output_image, prompt)
