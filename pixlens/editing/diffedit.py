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
    ) -> None:
        self.device = device
        self.model = load_diffedit(self.get_model_name(), device)

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
        }

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        del edit_info

        source_prompt, target_prompt = prompt.split("[SEP]")
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
        ).images[0]

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.DESCRIPTION

    def generate_prompt(self, edit: Edit) -> str:
        return generate_description_based_prompt(edit)
