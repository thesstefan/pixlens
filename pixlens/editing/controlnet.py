import enum

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

from pixlens.editing import interfaces
from pixlens.editing.utils import log_model_if_not_in_cache
from pixlens.utils import utils


class ControlNetType(enum.StrEnum):
    BASE = "lllyasviel/sd-controlnet-canny"


class StableDiffusionType(enum.StrEnum):
    BASE = "runwayml/stable-diffusion-v1-5"


def load_controlnet(
    model_type: ControlNetType,
    device: torch.device | None = None,
) -> StableDiffusionControlNetPipeline:
    pipe: StableDiffusionControlNetPipeline
    path_to_cache = utils.get_cache_dir()
    log_model_if_not_in_cache(model_type, path_to_cache)
    controlnet = ControlNetModel.from_pretrained(
        model_type,
        torch_dtype=torch.float16,
    )
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
        pix2pix_type: ControlNetType = ControlNetType.BASE,
        device: torch.device | None = None,
    ) -> None:
        self.device = device
        self.model = load_controlnet(pix2pix_type, device)

    def prepare_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path)
        image_array = np.array(image)
        image_array = cv2.Canny(image_array, 100, 200)
        image_array = image_array[:, :, None]
        image_array = np.concatenate(
            [image_array, image_array, image_array],
            axis=2,
        )
        return Image.fromarray(
            image_array.astype(np.uint8)
        )  # Convert ndarray back to Image

    def get_model_name(self) -> str:
        return "ControlNet"

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        *,
        num_inference_steps: int = 100,
        image_guidance_scale: float = 1.0,
    ) -> Image.Image:
        input_image = self.prepare_image(image_path)
        return self.model(
            prompt,
            input_image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
        ).images[0]
