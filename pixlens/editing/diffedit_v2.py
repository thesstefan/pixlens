import torch

from pixlens.editing import interfaces as editing_interfaces
from pixlens.editing.impl.diffedit.diffedit import diffedit
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
