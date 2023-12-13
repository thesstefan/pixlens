import logging

from PIL import Image
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


    def detect_with_owlvit(self, prompt: str, image_path: str):
        image = Image.open(image_path)
        inputs = self.owlvit_processor(text=[prompt], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.owlvit_model(**inputs)

        results = self.owlvit_processor.post_process_object_detection(outputs=outputs, threshold=self.detection_confidence_threshold, target_sizes=torch.Tensor([image.size[::-1]]).to(self.device))
        return results