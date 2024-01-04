import logging
import pathlib

from pixlens.editing import controlnet, diffedit, pix2pix
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.utils import yaml_constructible

logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(module)-30s | %(message)s",
    handlers=[logging.StreamHandler()],
)

NAME_TO_EDITING_MODEL: dict[str, type[PromptableImageEditingModel]] = {
    "ControlNet": controlnet.ControlNet,
    "Pix2pix": pix2pix.Pix2pix,
    "DiffEdit": diffedit.DiffEdit,
}


def load_editing_model_from_yaml(
    yaml_path: pathlib.Path | str,
) -> PromptableImageEditingModel:
    return yaml_constructible.load_class_from_yaml(
        yaml_path,
        NAME_TO_EDITING_MODEL,
    )
