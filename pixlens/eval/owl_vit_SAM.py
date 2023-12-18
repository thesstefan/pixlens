import logging

import torch

from pixlens.eval import owl_vit as eval_owl_vit
from pixlens.eval import sam as eval_sam
from pixlens.eval.detect_and_segment import PromptDetectAndBBoxSegmentModel


class OwlVitSam(PromptDetectAndBBoxSegmentModel):
    def __init__(
        self,
        owlvit_type: eval_owl_vit.OwlViTType = eval_owl_vit.OwlViTType.LARGE,
        sam_type: eval_sam.SAMType = eval_sam.SAMType.VIT_H,
        detection_confidence_threshold: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        logging.info(
            f"Loading OwlVitSam [OwlVitSam ({owlvit_type}) + SAM ({sam_type})]"
        )
        self.device = device
        sam_predictor = eval_sam.BBoxSamPredictor(sam_type, device=device)
        owlvit = eval_owl_vit.OwLViT(
            owlvit_type,
            device=device,
            detection_confidence_threshold=detection_confidence_threshold,
        )

        super(OwlVitSam, self).__init__(
            owlvit, sam_predictor, detection_confidence_threshold
        )

    def detect_and_segment(self, prompt: str, image_path: str):
        detection_output = self.promptable_detection_model.detect(prompt, image_path)
        segmentation_output = self.bbox_segmentation_model.segment(
            detection_output.bounding_boxes, image_path
        )
        return segmentation_output, detection_output
