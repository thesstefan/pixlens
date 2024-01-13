import enum

import requests
import torch
from omegaconf import OmegaConf
from PIL import Image

from pixlens.editing import interfaces as editing_interfaces
from pixlens.editing import utils as editing_utils
from pixlens.editing.impl.diffedit.diffedit import (
    diffedit,
    load_model_from_config,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.utils import utils


class DiffEditType(enum.StrEnum):
    BASE = "DiffEdit"


download_url_dict = {
    DiffEditType.BASE: "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
}

SUCCESS_CODE = 200
TIMEOUT = 10


class DiffEdit(editing_interfaces.PromptableImageEditingModel):
    device: torch.device | None

    def __init__(
        self,
        device: torch.device | None = None,
        ddim_steps: int = 80,
        diffedit_type: DiffEditType = DiffEditType.BASE,
        seed: int = 0,
        config_path: str = "pixlens/editing/impl/diffedit/configs/stable-diffusion/v1-inference.yaml",
    ) -> None:
        self.device = device
        self.diffedit_type = diffedit_type
        self.seed = seed
        self.ddim_steps = ddim_steps
        self.config_path = config_path
        self.load_model(config_path)

    def load_model(self, config_path: str) -> None:
        path_to_cache = utils.get_cache_dir()
        self.ckpt_path = (
            path_to_cache / "models--DiffEdit/v1-5-pruned-emaonly.ckpt"
        )
        config = OmegaConf.load(config_path)
        need_to_download = editing_utils.log_model_if_not_in_cache(
            "DiffEdit",
            path_to_cache,
        )
        if need_to_download:
            url = download_url_dict[self.diffedit_type]
            response = requests.get(url, stream=True, timeout=TIMEOUT)
            if response.status_code == SUCCESS_CODE:
                with self.ckpt_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
        self.model = load_model_from_config(config, self.ckpt_path)

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
            dst_prompt=target_prompt,
            ddim_steps=self.ddim_steps,
            seed=self.seed,
        )
        return images[0]
