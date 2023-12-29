import enum
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Owlv2ForObjectDetection,
    Owlv2Processor,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)

from pixlens.detection import interfaces
from pixlens.utils import utils
from pixlens.utils.utils import log_if_hugging_face_model_not_in_cache


class OwlViTType(enum.StrEnum):
    BASE32 = "google/owlvit-base-patch32"
    BASE16 = "google/owlvit-base-patch16"
    LARGE = "google/owlvit-large-patch14"
    BASE16_V2 = "google/owlv2-base-patch16"
    LARGE_V2 = "google/owlv2-large-patch14"


def load_owlvit(
    owlvit_type: OwlViTType,
    device: torch.device | None = None,
) -> tuple[
    Owlv2ForObjectDetection | OwlViTForObjectDetection,
    Owlv2Processor | OwlViTProcessor,
]:
    path_to_cache = utils.get_cache_dir()
    log_if_hugging_face_model_not_in_cache(owlvit_type, path_to_cache)

    model = (
        Owlv2ForObjectDetection
        if owlvit_type in [OwlViTType.BASE16_V2, OwlViTType.LARGE_V2]
        else OwlViTForObjectDetection
    ).from_pretrained(owlvit_type, cache_dir=path_to_cache)

    processor = (
        Owlv2Processor
        if owlvit_type in [OwlViTType.BASE16_V2, OwlViTType.LARGE_V2]
        else OwlViTProcessor
    ).from_pretrained(owlvit_type, cache_dir=path_to_cache)

    model.to(device)
    model.eval()

    # TODO: Fix this mypy error - there is probably something
    #       wrong with the OwlViT stubs, but I can't figure it out now.
    #
    #       The Self return value in the class methods doesn't seem to
    #       work correctly when inherited.
    return model, processor  # type: ignore[return-value]


class OwlViT(interfaces.PromptableDetectionModel):
    device: torch.device | None

    def __init__(
        self,
        owlvit_type: OwlViTType,
        device: torch.device | None = None,
        detection_confidence_threshold: float = 0.3,
    ) -> None:
        self.device = device
        self.model, self.processor = load_owlvit(owlvit_type, device)
        self.detection_confidence_threshold = detection_confidence_threshold

    def detect(
        self,
        prompt: str,
        image: Image.Image,
    ) -> interfaces.DetectionOutput:
        prompts = prompt.split(",")
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.detection_confidence_threshold,
            target_sizes=torch.Tensor([image.size[::-1]]).to(self.device),
        )[0]

        return interfaces.DetectionOutput(
            logits=results["scores"].cpu(),
            bounding_boxes=results["boxes"].cpu(),
            phrases=[prompts[idx] for idx in results["labels"]],
        )
