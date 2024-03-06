import logging
import pathlib
from importlib import metadata

from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.utils import yaml_constructible

# FIXME(thesstefan): This is required because NullTextInversion
# requires older versions of diffusers, while the other models need
# newer versions. Therefore, the available models are decided based on
# the used diffusers version.
#
# See https://github.com/thesstefan/pixlens/pull/54 for more details.
USE_OLD_DIFFUSERS_VERSION = metadata.version("diffusers") == "0.10.0"
NAME_TO_EDITING_MODEL: dict[str, type[PromptableImageEditingModel]]

if USE_OLD_DIFFUSERS_VERSION:
    from pixlens.editing import null_text_inversion

    NAME_TO_EDITING_MODEL = {
        "NullTextInversion": null_text_inversion.NullTextInversion,
    }
else:
    from pixlens.editing import (
        controlnet,
        diffedit,
        instruct_pix2pix,
        lcm,
        open_edit,
        vqgan_clip,
    )

    NAME_TO_EDITING_MODEL = {
        "ControlNet": controlnet.ControlNet,
        "InstructPix2Pix": instruct_pix2pix.InstructPix2Pix,
        "DiffEdit": diffedit.DiffEdit,
        "LCM": lcm.LCM,
        "OpenEdit": open_edit.OpenEdit,
        "VqGANClip": vqgan_clip.VqGANClip,
    }

logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(module)-30s | %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_editing_model_from_yaml(
    yaml_path: pathlib.Path | str,
) -> PromptableImageEditingModel:
    return yaml_constructible.load_class_from_yaml(
        yaml_path,
        NAME_TO_EDITING_MODEL,
    )
