from omegaconf import OmegaConf
import torch
from PIL import Image

from pixlens.editing import interfaces as editing_interfaces
from pixlens.editing.impl.diffedit.diffedit import (
    diffedit,
    load_model_from_config,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


class DiffEdit(editing_interfaces.PromptableImageEditingModel):
    device: torch.device | None

    def __init__(
        self,
        device: torch.device | None = None,
        ddim_steps: int = 80,
        seed: int = 0,
    ) -> None:
        self.device = device
        self.seed = seed
        self.ddim_steps = ddim_steps

    def load_model(self, config_path: str, ckpt_path: str) -> None:
        config = OmegaConf.load(config_path)
        self.model = load_model_from_config(config, ckpt_path)

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "ddim_steps": self.ddim_steps,
            "seed": self.seed,
        }

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        edit_info: Edit | None = None,
    ) -> Image.Image:
        del edit_info

        source_prompt, target_prompt = prompt.split("[SEP]")
        images, _ = diffedit(
            self.model,
            image_path,
            src_prompt=source_prompt,
            tgt_prompt=target_prompt,
            ddim_steps=self.ddim_steps,
            seed=self.seed,
        )
        return images[0]
