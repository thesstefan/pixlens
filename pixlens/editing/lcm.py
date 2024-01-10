import enum

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

from pixlens.editing import interfaces
from pixlens.editing.utils import (
    generate_instruction_based_prompt,
    log_model_if_not_in_cache,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


class LCMType(enum.StrEnum):
    BASE = "SimianLuo/LCM_Dreamshaper_v7"


def load_lcm(
    model_type: LCMType = LCMType.BASE,
    device: torch.device | None = None,
) -> AutoPipelineForImage2Image:
    path_to_cache = utils.get_cache_dir()
    log_model_if_not_in_cache(model_type, path_to_cache)
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_type,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=path_to_cache,
    )

    pipe.to(device)
    pipeline: AutoPipelineForImage2Image = pipe
    return pipeline


class LCM(interfaces.PromptableImageEditingModel):
    model: AutoPipelineForImage2Image
    lcm_type: LCMType
    device: torch.device | None
    num_inference_steps: int
    image_guidance_scale: float
    text_guidance_scale: float

    def __init__(  # noqa: PLR0913
        self,
        lcm_type: LCMType = LCMType.BASE,
        device: torch.device | None = None,
        num_inference_steps: int = 10,
        image_guidance_scale: float = 1.0,
        text_guidance_scale: float = 7.5,
    ) -> None:
        self.device = device
        self.model = load_lcm(lcm_type, device)
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.text_guidance_scale = text_guidance_scale

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "lcm_type": str(self.lcm_type),
            "num_inference_steps": self.num_inference_steps,
            "image_guidance_scale": self.image_guidance_scale,
            "text_guidance_scale": self.text_guidance_scale,
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
            guidance_scale=self.text_guidance_scale,
        ).images[0]

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.INSTRUCTION

    def generate_prompt(self, edit: Edit) -> str:
        return generate_instruction_based_prompt(edit)
