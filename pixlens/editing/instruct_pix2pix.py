import enum

import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from PIL import Image

from pixlens.editing import interfaces
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


class InstructPix2PixType(enum.StrEnum):
    BASE = "timbrooks/instruct-pix2pix"


def load_instruct_pix2pix(
    model_type: InstructPix2PixType,
    device: torch.device | None = None,
) -> StableDiffusionInstructPix2PixPipeline:
    cache_dir = utils.get_cache_dir()
    utils.log_if_hugging_face_model_not_in_cache(
        model_type,
        cache_dir,
    )
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_type,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=cache_dir,
    )

    pipe.to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
    )
    pipeline: StableDiffusionInstructPix2PixPipeline = pipe
    return pipeline


class InstructPix2Pix(interfaces.PromptableImageEditingModel):
    instruct_pix2pix_type: InstructPix2PixType
    device: torch.device | None
    model: StableDiffusionInstructPix2PixPipeline
    num_inference_steps: int
    image_guidance_scale: float
    text_guidance_scale: float

    def __init__(  # noqa: PLR0913
        self,
        instruct_pix2pix_type: InstructPix2PixType = InstructPix2PixType.BASE,
        device: torch.device | None = None,
        num_inference_steps: int = 100,
        image_guidance_scale: float = 1.0,
        text_guidance_scale: float = 7.5,
        seed: int = 0,
        latent_guidance_scale: float = 25,
    ) -> None:
        self.model = load_instruct_pix2pix(instruct_pix2pix_type, device)
        self.instruct_pix2pix_type = instruct_pix2pix_type
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale
        self.latent_guidance_scale = latent_guidance_scale
        self.seed = seed

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "instruct_pix2pix_type": str(self.instruct_pix2pix_type),
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
        del edit_info

        input_image = Image.open(image_path)
        return self.model(  # type: ignore[operator, no-any-return]
            prompt,
            input_image,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
            generator=torch.manual_seed(self.seed),
            guidance_scale=self.text_guidance_scale,
        ).images[0]

    def get_latent(self, prompt: str, image_path: str) -> torch.Tensor:
        input_image = Image.open(image_path)

        output = self.model(
            prompt,
            input_image,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
            output_type="latent",
            generator=torch.manual_seed(self.seed),
            guidance_scale=self.latent_guidance_scale,
        )
        output_images: list[torch.Tensor] = output.images
        return output_images[0]

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.INSTRUCTION
