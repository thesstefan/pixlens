import logging

import torch

from pixlens.detection import interfaces
from pixlens.detection.owl_vit import OwlViT, OwlViTType
from pixlens.detection.sam import BBoxSamPredictor, SAMType


class OwlViTSAM(interfaces.PromptDetectAndBBoxSegmentModel):
    device: torch.device | None

    sam_type: SAMType
    owlvit_type: OwlViTType

    detection_confidence_threshold: float

    def __init__(
        self,
        owlvit_type: OwlViTType = OwlViTType.LARGE,
        sam_type: SAMType = SAMType.VIT_H,
        detection_confidence_threshold: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        logging.info(
            "Loading OwlViT+SAM [OwlViT (%s) + SAM (%s)]",
            owlvit_type,
            sam_type,
        )
        self.device = device
        self.sam_type = sam_type
        self.owlvit_type = owlvit_type
        self.detection_confidence_threshold = detection_confidence_threshold

        sam_predictor = BBoxSamPredictor(sam_type, device=device)
        owlvit = OwlViT(
            owlvit_type,
            device=device,
            detection_confidence_threshold=detection_confidence_threshold,
        )

        super().__init__(
            owlvit,
            sam_predictor,
            detection_confidence_threshold,
        )

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "owlvit_type": str(self.owlvit_type),
            "sam_type": str(self.sam_type),
            "detection_confidence_threshold": (
                self.detection_confidence_threshold
            ),
        }
