import torch
from diffusers.schedulers import DDIMScheduler
from PIL import Image

from pixlens.dataset.prompt_utils import split_description_based_prompt
from pixlens.editing import interfaces
from pixlens.editing.impl.null_text_inversion.hf_pipeline import (
    NullTextPipeline,
)
from pixlens.editing.stable_diffusion import StableDiffusionType
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


def load_nulltext_inversion(  # type: ignore[no-any-unimported]
    sd_type: StableDiffusionType,
    device: torch.device | None,
) -> NullTextPipeline:
    cache_dir = utils.get_cache_dir()
    utils.log_if_hugging_face_model_not_in_cache(sd_type, cache_dir)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
        beta_schedule="scaled_linear",
    )

    return NullTextPipeline.from_pretrained(  # type: ignore[no-any-return]
        sd_type,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    ).to(device)


class NullTextInversion(interfaces.PromptableImageEditingModel):
    null_inversion: NullTextPipeline

    device: torch.device | None

    sd_type: StableDiffusionType
    num_inner_steps: int
    num_steps: int
    guidance_scale: float

    def __init__(  # noqa: PLR0913
        self,
        sd_type: StableDiffusionType = StableDiffusionType.V15,
        num_inner_steps: int = 10,
        num_steps: int = 10,
        guidance_scale: float = 7.5,
        early_stop_epsilon: float = 1e-5,
        device: torch.device | None = None,
    ) -> None:
        self.device = device
        self.sd_type = sd_type

        self.num_inner_steps = num_inner_steps
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.early_stop_epsilon = early_stop_epsilon

        self.null_inversion = load_nulltext_inversion(sd_type, device)

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "sd_type": str(self.sd_type),
            "num_steps": self.num_steps,
            "num_inner_steps": self.num_inner_steps,
            "guidance_scale": self.guidance_scale,
        }

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        del edit_info

        invert_prompt, prompt = split_description_based_prompt(prompt)

        inverted_latent, uncond_embeddings = self.null_inversion.invert(
            image_path,
            invert_prompt,
            num_inner_steps=self.num_inner_steps,
            num_inference_steps=self.num_steps,
            early_stop_epsilon=self.early_stop_epsilon,
        )

        return self.null_inversion(  # type: ignore[no-any-return]
            prompt,
            uncond_embeddings,
            inverted_latent,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_steps,
        ).images[0]  # type: ignore[no-any-return]

    def get_latent(self, prompt: str, image_path: str) -> torch.Tensor:
        invert_prompt, prompt = split_description_based_prompt(prompt)

        inverted_latent, uncond_embeddings = self.null_inversion.invert(
            image_path,
            invert_prompt,
            num_inner_steps=self.num_inner_steps,
            num_inference_steps=self.num_steps,
            early_stop_epsilon=self.early_stop_epsilon,
        )

        # TODO: Can we do better by using some info from the inverted latent?
        return self.null_inversion(  # type: ignore[no-any-return]
            prompt,
            uncond_embeddings,
            inverted_latent,
            output_type="latent",
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_steps,
        ).images[0]

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.DESCRIPTION
