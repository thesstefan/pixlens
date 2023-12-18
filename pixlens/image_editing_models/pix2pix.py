import PIL
import logging
import enum
import torch
from pathlib import Path
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)

from pixlens import utils
from pixlens.eval import interfaces


class Pix2pixType(enum.StrEnum):
    BASE = "timbrooks/instruct-pix2pix"


def log_model_if_not_in_cache(model_name: str, cache_dir: Path) -> None:
    model_dir = model_name.replace("/", "--")
    model_dir = "models--" + model_dir
    full_path = cache_dir / model_dir
    if not full_path.is_dir():
        logging.info(f"Downloading Pix2pix model from {model_name}...")


def load_pix2pix(
    model_type: Pix2pixType, device: torch.device | None = None
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
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


class Pix2pix(interfaces.PromptableImageEditingModel):
    device: torch.device | None

    def __init__(
        self,
        pix2pix_type: Pix2pixType,
        device: torch.device | None = None,
        *,
        num_inference_steps: int = 100,
        image_guidance_scale: float = 1.0,
    ) -> None:
        self.device = device
        self.model = load_pix2pix(pix2pix_type, device)
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale

    def edit_image(
        self,
        prompt: str,
        image_path: str,
    ) -> interfaces.ImageEditingOutput:
        input_image = PIL.Image.open(image_path)
        output_image = self.model(
            prompt,
            input_image,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
        ).images[0]
        return interfaces.ImageEditingOutput(output_image, prompt)
