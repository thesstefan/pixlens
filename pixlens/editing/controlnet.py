import enum

import cv2  # type: ignore[import]
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline,
)
from PIL import Image

from pixlens.editing import interfaces
from pixlens.editing.stable_diffusion import StableDiffusionType
from pixlens.editing.utils import (
    generate_instruction_based_prompt,
    log_model_if_not_in_cache,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


class ControlNetType(enum.StrEnum):
    CANNY = "lllyasviel/sd-controlnet-canny"


def load_controlnet(
    model_type: ControlNetType,
    device: torch.device | None = None,
) -> StableDiffusionControlNetPipeline:
    path_to_cache = utils.get_cache_dir()
    log_model_if_not_in_cache(model_type, path_to_cache)
    controlnet = ControlNetModel.from_pretrained(
        model_type,
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        StableDiffusionType.V15,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=path_to_cache,
    )

    pipe.to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipeline: StableDiffusionControlNetPipeline = pipe
    return pipeline


class ControlNet(interfaces.PromptableImageEditingModel):
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

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "controlnet_type": str(self.controlnet_type),
            "num_inference_steps": self.num_inference_steps,
            "image_guidance_scale": self.image_guidance_scale,
            "text_guidance_scale": self.text_guidance_scale,
        }

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
            image_array.astype(np.uint8),
        )  # Convert ndarray back to Image

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        del edit_info
        input_image = self.prepare_image(image_path)
        return self.model(  # type: ignore[no-any-return]
            prompt,
            input_image,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
            guidance_scale=self.text_guidance_scale,
            generator=torch.manual_seed(self.seed),
        ).images[0]

    def get_latent(self, prompt: str, image_path: str) -> torch.Tensor:
        input_image = self.prepare_image(image_path)
        return self.model(  # type: ignore[no-any-return]
            prompt,
            input_image,
            num_inference_steps=100,
            image_guidance_scale=1.0,
            output_type="latent",
            guidance_scale=self.latent_guidance_scale,
            generator=torch.manual_seed(self.seed),
        ).images[0]

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.INSTRUCTION

    def generate_prompt(self, edit: Edit) -> str:
        return generate_instruction_based_prompt(edit)
