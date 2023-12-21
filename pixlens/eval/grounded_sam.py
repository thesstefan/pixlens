import logging

import torch

from pixlens.eval import grounding_dino as eval_grounding_dino
from pixlens.eval import sam as eval_sam


class GroundedSAM(interfaces.PromptDetectAndBBoxSegmentModel):
    def __init__(
        self,
        sam_type: eval_sam.SAMType = eval_sam.SAMType.VIT_H,
        grounding_dino_type: eval_grounding_dino.GroundingDINOType = (
            eval_grounding_dino.GroundingDINOType.SWIN_T
        ),
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
