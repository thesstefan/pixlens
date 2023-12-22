import enum
from pathlib import Path

import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    BlipModel,
)


from pixlens.detection.automatic_label.interfaces import (
    ImageDescriptorModel,
    ImageDescription,
)
from pixlens.detection.utils import log_if_hugging_face_model_not_in_cache
from pixlens.utils import get_cache_dir


class BlipType(enum.StrEnum):
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    BLIP1 = "Salesforce/blip-image-captioning-base"


def load_blip(
    blip_type: BlipType,
    device: torch.device | None = None,
) -> tuple[
    Blip2ForConditionalGeneration | BlipModel,
    Blip2Processor | AutoProcessor,
]:
    log_if_hugging_face_model_not_in_cache(blip_type)
    path_to_cache = get_cache_dir()
    model = (
        Blip2ForConditionalGeneration
        if blip_type in [BlipType.BLIP2]
        else BlipModel
    ).from_pretrained(blip_type, cache_dir=path_to_cache)
    processor = (
        Blip2Processor if blip_type in [BlipType.BLIP2] else AutoProcessor
    ).from_pretrained(blip_type, cache_dir=path_to_cache)

    model.to(device)
    model.eval()
    return model, processor


class BlipModel(ImageDescriptorModel):
    device: torch.device | None

    def __init__(
        self,
        blip_type: BlipType = BlipType.BLIP2,
        device: torch.device | None = None,
    ) -> None:
        self.device = device
