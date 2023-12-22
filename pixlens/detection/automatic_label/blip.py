import enum
import logging
from pathlib import Path

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from pixlens.detection.automatic_label.interfaces import (
    ImageDescriptorModel,
    ImageDescription,
)


class BlipType(enum.StrEnum):
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    BLIP1 = "Salesforce/blip-image-captioning-base"


def log_if_model_not_in_cache(blip_type: str, cache_dir: Path) -> None:
    model_dir = blip_type.replace("/", "--")
    model_dir = "models--" + model_dir

    if not (cache_dir / model_dir).is_dir():
        logging.info("Downloading Blip %s...", blip_type)


class BlipModel(ImageDescriptorModel):
    device: torch.device | None

    def __init__(
        self,
        blip_type: BlipType = BlipType.BLIP2,
        device: torch.device | None = None,
    ) -> None:
        self.device = device
