import logging

import torch

from pixlens.detection import interfaces
from pixlens.detection import owl_vit as eval_owl_vit
from pixlens.detection import sam as eval_sam


class OwlViTSAM(interfaces.PromptDetectAndBBoxSegmentModel):
    def __init__(
        self,
        owlvit_type: eval_owl_vit.OwlViTType = eval_owl_vit.OwlViTType.LARGE,
        sam_type: eval_sam.SAMType = eval_sam.SAMType.VIT_H,
        detection_confidence_threshold: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        logging.info(
            "Loading OwlViT+SAM [OwlViT (%s) + SAM (%s)]",
            owlvit_type,
            sam_type,
        )
        self.device = device
        sam_predictor = eval_sam.BBoxSamPredictor(sam_type, device=device)
        owlvit = eval_owl_vit.OwlViT(
            owlvit_type,
            device=device,
            detection_confidence_threshold=detection_confidence_threshold,
        )

        super().__init__(
            owlvit,
            sam_predictor,
            detection_confidence_threshold,
        )
