import logging

import torch

from pixlens.detection import grounding_dino as eval_grounding_dino
from pixlens.detection import interfaces
from pixlens.detection import sam as eval_sam
from pixlens.utils.yaml_constructible import YamlConstructible


class GroundedSAM(
    interfaces.PromptDetectAndBBoxSegmentModel,
    YamlConstructible,
):
    def __init__(
        self,
        grounding_dino_type: eval_grounding_dino.GroundingDINOType = (
            eval_grounding_dino.GroundingDINOType.SWIN_T
        ),
        sam_type: eval_sam.SAMType = eval_sam.SAMType.VIT_H,
        detection_confidence_threshold: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        logging.info(
            "Loading GroundedSAM [GroundingDINO (%s) + SAM (%s)]",
            grounding_dino_type,
            sam_type,
        )

        sam_predictor = eval_sam.BBoxSamPredictor(sam_type, device=device)
        grounding_dino = eval_grounding_dino.GroundingDINO(
            grounding_dino_type,
            device=device,
        )

        super().__init__(
            grounding_dino,
            sam_predictor,
            detection_confidence_threshold,
        )
