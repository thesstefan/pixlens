import logging

import torch

from pixlens.eval import owl_vit as eval_owl_vit
from pixlens.eval import interfaces
from pixlens.eval import sam as eval_sam
from pixlens.eval.grounded_sam import PromptDetectAndBBoxSegmentModel


class OwlVitSam(PromptDetectAndBBoxSegmentModel):
    def __init__(
        self,
        owlvit_model_type: eval_owl_vit.OwlViTType = eval_owl_vit.OwlViTType.base32,
        sam_type: eval_sam.SAMType = eval_sam.SAMType.VIT_H,
        detection_confidence_threshold: float = 0.3,
        device: str = 'cpu'
    ) -> None:
        logging.info(
            f"Loading OwlVitSam [OwlVitSam ({owlvit_model_type}) + SAM ({sam_type})]"
        )

        sam_predictor = eval_sam.BBoxSamPredictor(sam_type, device=device)
        model_owlvit, self.processor_owlvit = eval_owl_vit.load_owlvit(owlvit_model_type, device=device)


        super(OwlVitSam, self).__init__(
            model_owlvit, sam_predictor, detection_confidence_threshold
        )


