import enum
import logging
import pathlib

import numpy.typing as npt
import torch
from typing import Any
from groundingdino.util import box_ops, inference

from pixlens import utils
from pixlens.eval import interfaces


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
GROUNDING_DINO_CONFIG_NAMES = utils.get_basename_dict(
    GROUNDING_DINO_CONFIG_URLS,
)


def get_grounding_dino_ckpt(
    grounding_dino_type: GroundingDINOType,
) -> pathlib.Path:
    ckpt_path = (
        utils.get_cache_dir() / GROUNDING_DINO_CKPT_NAMES[grounding_dino_type]
    )

    if not ckpt_path.exists():
        logging.info(
            "Downloading GroundingDINO (%s) weights from %s...",
            grounding_dino_type,
            GROUNDING_DINO_CKPT_URLS[grounding_dino_type],
        )
        utils.download_file(
            GROUNDING_DINO_CKPT_URLS[grounding_dino_type],
            ckpt_path,
            desc=f"GroundingDINO {grounding_dino_type}",
        )

    return ckpt_path


def get_grounding_dino_config(
    grounding_dino_type: GroundingDINOType,
) -> pathlib.Path:
    config_path = (
        utils.get_cache_dir() / GROUNDING_DINO_CONFIG_NAMES[grounding_dino_type]
    )

    if not config_path.exists():
        logging.info(
            "Downloading GroundingDINO (%s) config from %s...",
            grounding_dino_type,
            GROUNDING_DINO_CONFIG_URLS[grounding_dino_type],
        )
        utils.download_text_file(
            GROUNDING_DINO_CONFIG_URLS[grounding_dino_type],
            config_path,
        )

    return config_path


def load_grounding_dino(
    grounding_dino_type: GroundingDINOType,
    device: torch.device | None = None,
) -> torch.nn.Module:
    model_ckpt = get_grounding_dino_ckpt(grounding_dino_type)
    model_config = get_grounding_dino_config(grounding_dino_type)

    logging.info(
        "Loading GroundingDINO %s from %s...",
        grounding_dino_type,
        model_ckpt,
    )

    return inference.load_model(
        str(model_config),
        str(model_ckpt),
        device=str(device),
    )


class GroundingDINO(interfaces.PromptableDetectionModel):
    grounding_dino_model: torch.nn.Module

    box_threshold: float
    text_threshold: float

    device: torch.device | None

    def __init__(
        self,
        grounding_dino_type: GroundingDINOType,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        self.grounding_dino_model = load_grounding_dino(
            grounding_dino_type,
            device,
        )
        self.device = device

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def _unnormalize_bboxes(
        self,
        bboxes: torch.Tensor,
        image: npt.NDArray[Any],
    ) -> torch.Tensor:
        height, width, _ = image.shape
        return box_ops.box_cxcywh_to_xyxy(bboxes) * torch.Tensor(
            [width, height, width, height],
        )

    def detect(
        self,
        prompt: str,
        image_path: str,
    ) -> interfaces.DetectionOutput:
        image_source, image = inference.load_image(image_path)

        boxes, logits, phrases = inference.predict(
            model=self.grounding_dino_model,
            image=image,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=str(self.device),
        )

        return interfaces.DetectionOutput(
            self._unnormalize_bboxes(boxes, image_source),
            logits,
            phrases,
        )
