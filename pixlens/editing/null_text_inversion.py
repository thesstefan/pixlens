import torch
from diffusers import StableDiffusionPipeline  # type: ignore[import]
from PIL import Image

from pixlens.editing import interfaces
from pixlens.editing import utils as editing_utils
from pixlens.editing.impl.null_text_inversion import (
    controllers,
    null_inversion,
    ptp_utils,
)
from pixlens.editing.stable_diffusion import StableDiffusionType
from pixlens.evaluation.interfaces import Edit, EditType
from pixlens.utils import utils


def load_stable_diffusion(  # type: ignore[no-any-unimported]
    sd_type: StableDiffusionType,
    device: torch.device | None,
) -> StableDiffusionPipeline:
    cache_dir = utils.get_cache_dir()
    utils.log_if_hugging_face_model_not_in_cache(sd_type, cache_dir)

    return StableDiffusionPipeline.from_pretrained(  # type: ignore[no-any-return]
        sd_type,
        cache_dir=cache_dir,
    ).to(device)


class NullTextInversion(interfaces.PromptableImageEditingModel):
    null_inversion: null_inversion.NullInversion
    ldm_stable: StableDiffusionPipeline

    device: torch.device | None
    generator: torch.Generator

    sd_type: StableDiffusionType
    seed: int
    num_ddim_steps: int
    guidance_scale: float
    cross_replace_steps: float
    self_replace_steps: float
    subject_amplification: float

    def __init__(  # noqa: PLR0913
        self,
        seed: int,
        sd_type: StableDiffusionType = StableDiffusionType.V14,
        num_ddim_steps: int = 50,
        guidance_scale: float = 7.5,
        cross_replace_steps: float = 0.8,
        self_replace_steps: float = 0.5,
        subject_amplification: float = 2.0,
        device: torch.device | None = None,
    ) -> None:
        self.device = device

        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale

        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.subject_amplification = subject_amplification

        self.ldm_stable = load_stable_diffusion(sd_type, device)

        self.null_inversion = null_inversion.NullInversion(
            self.ldm_stable,
            self.num_ddim_steps,
            self.guidance_scale,
            self.device,
        )

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "sd_type": str(self.sd_type),
            "seed": self.seed,
            "num_ddim_steps": self.num_ddim_steps,
            "guidance_scale": self.guidance_scale,
            "cross_replace_steps": self.cross_replace_steps,
            "self_replace_steps": self.self_replace_steps,
            "subject_amplification": self.subject_amplification,
        }

    def get_inversion_latent(
        self,
        img_path: str,
        source_prompt: str,
    ) -> tuple[torch.FloatTensor, torch.Tensor]:
        _, x_t, uncond_embeddings = self.null_inversion.invert(
            img_path,
            source_prompt,
            offsets=(0, 0, 0, 0),
            verbose=True,
        )

        controller = controllers.AttentionStore()
        _, x_t = ptp_utils.text2image_ldm_stable(
            self.ldm_stable,
            [source_prompt],
            controller,
            latent=x_t,
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
            uncond_embeddings=uncond_embeddings,
        )

        return x_t, uncond_embeddings

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        if not edit_info:
            msg = "edit_info is required for Null-Text Inversion edit"
            raise utils.GotNoneError(msg)

        src, dst = editing_utils.split_description_based_prompt(prompt)
        x_t, uncond_embeddings = self.get_inversion_latent(image_path, src)

        # TODO(thesstefan): Play more with these parameters and see what
        #                   can be achieved with them. May need to make them
        #                   more operation specific!
        equalizer_params = {
            "words": (edit_info.to_attribute,),
            "values": (self.subject_amplification,),
        }
        blend_words = (
            (edit_info.to_attribute,),
            (
                edit_info.to_attribute
                if edit_info.edit_type == EditType.OBJECT_REPLACEMENT
                else edit_info.from_attribute,
            ),
        )

        # FIXME(thesstefan): Using the blend_words parameter
        # makes the implementation code raise a KeyError. This
        # seems to be because of our pinned diffusers version.
        # This parameters seems to be quite important since it
        # defines "locality" in the edit.
        #
        # Here are some related issues that I found:
        #   https://github.com/google/prompt-to-prompt/issues/57
        #   https://github.com/google/prompt-to-prompt/issues/72
        #   https://github.com/google/prompt-to-prompt/issues/37
        #
        # Pinning the diffusers version to something earlier like
        # 0.10.0 makes it work. I can't reproduce the results
        # from the paper though. We can't downgrade because we need
        # other things from newer versions. For example ControlNet
        # is introduced in 0.25.0.
        #
        # We have two choices:
        #   - Try to run this model in a separate environment
        #   and remove the other dependencies e.g. ControlNet that
        #   use a newer version of version.
        #
        #   - Wait for an official HF version to be implemented.
        #   There seems to be some recent work being done on it (8 days ago):
        #       https://github.com/huggingface/diffusers/issues/6313
        controller = controllers.make_controller(
            [src, dst],
            self.ldm_stable.tokenizer,  # type: ignore[attr-defined]
            edit_info.edit_type == EditType.OBJECT_REPLACEMENT,
            {"default_": self.cross_replace_steps},
            self.self_replace_steps,
            num_ddim_steps=50,
            blend_words=blend_words,
            equalizer_params=equalizer_params,
        )

        images, _ = ptp_utils.text2image_ldm_stable(
            self.ldm_stable,
            [src, dst],
            controller,
            latent=x_t,
            uncond_embeddings=uncond_embeddings,
            num_inference_steps=self.num_ddim_steps,
            guidance_scale=self.guidance_scale,
        )

        return Image.fromarray(images[1])

    @property
    def prompt_type(self) -> interfaces.ImageEditingPromptType:
        return interfaces.ImageEditingPromptType.DESCRIPTION

    def generate_prompt(self, edit: Edit) -> str:
        return editing_utils.generate_description_based_prompt(edit)
