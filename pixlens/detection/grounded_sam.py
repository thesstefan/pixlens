import logging

import torch

from pixlens.detection import interfaces
from pixlens.detection.grounding_dino import GroundingDINO, GroundingDINOType
from pixlens.detection.sam import BBoxSamPredictor, SAMType


class GroundedSAM(interfaces.PromptDetectAndBBoxSegmentModel):
    grounding_dino_type: GroundingDINOType
    sam_type: SAMType
    detection_confidence_threshold: float
    device: torch.device | None

    def __init__(
        self,
        grounding_dino_type: GroundingDINOType = GroundingDINOType.SWIN_T,
        sam_type: SAMType = SAMType.VIT_H,
        detection_confidence_threshold: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        self.grounding_dino_type = grounding_dino_type
        self.sam_type = sam_type
        self.detection_confidence_threshold = detection_confidence_threshold
        self.device = device

        logging.info(
            "Loading GroundedSAM [GroundingDINO (%s) + SAM (%s)]",
            grounding_dino_type,
            sam_type,
        )

        sam_predictor = BBoxSamPredictor(sam_type, device=device)
        grounding_dino = GroundingDINO(
            grounding_dino_type,
            device=device,
        )

        super().__init__(
            grounding_dino,
            sam_predictor,
            detection_confidence_threshold,
        )

    @property
    def params_dict(self) -> dict[str, str | bool | int | float]:
        return {
            "device": str(self.device),
            "grounding_dino_type": str(self.grounding_dino_type),
            "sam_type": str(self.sam_type),
            "detection_confidence_threshold": (
                self.detection_confidence_threshold
            ),
        }
