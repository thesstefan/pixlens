import numpy as np
import torch
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionDiffEditPipeline,
)
from PIL import Image

from pixlens.editing import interfaces
from pixlens.editing.stable_diffusion import StableDiffusionType
from pixlens.editing.utils import (
    generate_description_based_prompt,
    log_model_if_not_in_cache,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


def load_diffedit(
    model_name: str,
    device: torch.device | None = None,
) -> StableDiffusionDiffEditPipeline:
    path_to_cache = utils.get_cache_dir()
    log_model_if_not_in_cache(model_name, path_to_cache)
    pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
        StableDiffusionType.V21,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True,
    )
    pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(
        pipeline.scheduler.config,
    )
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()
    return pipeline  # type: ignore[no-any-return]


class DiffEdit(interfaces.PromptableImageEditingModel):
    device: torch.device | None

    def __init__(
        self,
        device: torch.device | None = None,
        latent_guidance_scale: float = 25.0,
        seed: int = 0,
    ) -> None:
        self.device = device
        self.model = load_diffedit(self.get_model_name(), device)
        self.latent_guidance_scale = latent_guidance_scale
        self.seed = seed

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "latent_guidance_scale": self.latent_guidance_scale,
            "seed": self.seed,
        }

    def resize_image(self, image: Image.Image) -> torch.Tensor:
        # Editval is 512,512
        w, h = image.size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        np_image = np.array(image).astype(np.float32) / 255.0
        np_image = np_image[None].transpose(0, 3, 1, 2)
        image_torch = torch.from_numpy(np_image)
        return 2.0 * image_torch - 1.0

    def simple_resize_image(self, image: Image.Image) -> Image.Image:
        # w, h = image.size
        # w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        w, h = 512, 512
        return image.resize((w, h), resample=Image.Resampling.LANCZOS)

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        del edit_info

        source_prompt, target_prompt = prompt.split("[SEP]")
        input_image = Image.open(image_path)
        input_image = self.simple_resize_image(input_image)
        mask_image = self.model.generate_mask(  # type: ignore[attr-defined]
            image=input_image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
        )
        inv_latents = self.model.invert(  # type: ignore[attr-defined]
            prompt=source_prompt,
            image=input_image,
            num_inference_steps=100,
        ).latents

        return self.model(  # type: ignore[operator, no-any-return]
            prompt=target_prompt,
            mask_image=mask_image,
            image_latents=inv_latents,
            negative_prompt=source_prompt,
            generator=torch.manual_seed(self.seed),
            num_inference_steps=100,
        ).images[0]

    def get_latent(self, prompt: str, image_path: str) -> torch.Tensor:
        source_prompt, target_prompt = prompt.split("[SEP]")
        source_prompt = source_prompt[13:]
        target_prompt = target_prompt[13:]
        input_image = Image.open(image_path)
        mask_image = self.model.generate_mask(  # type: ignore[attr-defined]
            image=input_image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
        )
        inv_latents = self.model.invert(  # type: ignore[attr-defined]
            prompt=source_prompt,
            image=input_image,
        ).latents

        return self.model(  # type: ignore[operator, no-any-return]
            prompt=target_prompt,
            mask_image=mask_image,
            image_latents=inv_latents,
            negative_prompt=source_prompt,
            output_type="latent",
            guidance_scale=self.latent_guidance_scale,
            generator=torch.manual_seed(self.seed),
        ).images[0]

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.DESCRIPTION

    def generate_prompt(self, edit: Edit) -> str:
        return generate_description_based_prompt(edit)
