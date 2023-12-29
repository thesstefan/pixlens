import enum

import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from PIL import Image

from pixlens.editing import interfaces
from pixlens.editing.utils import log_model_if_not_in_cache
from pixlens.utils import utils
from pixlens.utils.yaml_constructible import YamlConstructible


class Pix2pixType(enum.StrEnum):
    BASE = "timbrooks/instruct-pix2pix"


def load_pix2pix(
    model_type: Pix2pixType,
    device: torch.device | None = None,
) -> StableDiffusionInstructPix2PixPipeline:
    path_to_cache = utils.get_cache_dir()
    log_model_if_not_in_cache(model_type, path_to_cache)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_type,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=path_to_cache,
    )

    pipe.to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
    )
    pipeline: StableDiffusionInstructPix2PixPipeline = pipe
    return pipeline


class Pix2pix(
    interfaces.PromptableImageEditingModel,
    YamlConstructible,
):
    device: torch.device | None

    def __init__(
        self,
        pix2pix_type: Pix2pixType = Pix2pixType.BASE,
        device: torch.device | None = None,
        num_inference_steps: int = 100,
        image_guidance_scale: float = 1.0,
        text_guidance_scale: float = 7.5,
    ) -> None:
        self.device = device
        self.model = load_pix2pix(pix2pix_type, device)
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale

    def get_model_name(self) -> str:
        return "Pix2pix"

    def edit_image(
        self,
        prompt: str,
        image_path: str,
    ) -> Image.Image:
        input_image = Image.open(image_path)
        output = self.model(
            prompt,
            input_image,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
            guidance_scale=self.text_guidance_scale,
        )  # TODO: controlnet this is not detected as a mistake.
        output_images: list[Image.Image] = output.images
        return output_images[0]
