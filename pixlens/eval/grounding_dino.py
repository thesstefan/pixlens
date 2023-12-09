import enum
import logging
import pathlib

from groundingdino import models
from groundingdino.util import inference

from pixlens import utils


class GroundingDINOType(enum.StrEnum):
    SWIN_T = "swin_t"
    SWIN_B = "swin_b"


GROUNDING_DINO_CKPT_URLS = {
    GroundingDINOType.SWIN_T: "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth",
    GroundingDINOType.SWIN_B: "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
}
GROUNDING_DINO_CKPT_NAMES = utils.get_basename_dict(GROUNDING_DINO_CKPT_URLS)

GROUNDING_DINO_CONFIG_URLS = {
    GroundingDINOType.SWIN_T: "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/"
    "main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    GroundingDINOType.SWIN_B: "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/"
    "main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
}
GROUNDING_DINO_CONFIG_NAMES = utils.get_basename_dict(GROUNDING_DINO_CONFIG_URLS)


def get_grounding_dino_ckpt(grounding_dino_type: GroundingDINOType) -> pathlib.Path:
    ckpt_path = utils.get_cache_dir() / GROUNDING_DINO_CKPT_NAMES[grounding_dino_type]

    if not ckpt_path.exists():
        logging.info(
            f"Downloading GroundingDINO ({grounding_dino_type}) weights from "
            f"{GROUNDING_DINO_CKPT_URLS[grounding_dino_type]}..."
        )
        utils.download_file(
            GROUNDING_DINO_CKPT_URLS[grounding_dino_type],
            ckpt_path,
            desc=f"GroundingDINO {grounding_dino_type}",
        )

    return ckpt_path


def get_grounding_dino_config(grounding_dino_type: GroundingDINOType) -> pathlib.Path:
    config_path = (
        utils.get_cache_dir() / GROUNDING_DINO_CONFIG_NAMES[grounding_dino_type]
    )

    if not config_path.exists():
        logging.info(
            f"Downloading GroundingDINO ({grounding_dino_type}) config from "
            f"{GROUNDING_DINO_CONFIG_URLS[grounding_dino_type]}"
        )
        utils.download_file(
            GROUNDING_DINO_CONFIG_URLS[grounding_dino_type], config_path, text=True
        )

    return config_path


def load_grounding_dino(
    grounding_dino_type: GroundingDINOType = GroundingDINOType.SWIN_T,
) -> models.GroundingDINO.groundingdino.GroundingDINO:
    model_ckpt = get_grounding_dino_ckpt(grounding_dino_type)
    model_config = get_grounding_dino_config(grounding_dino_type)

    logging.info(f"Loading GroundingDINO {grounding_dino_type} from {model_ckpt}...")

    return inference.load_model(str(model_config), str(model_ckpt))
