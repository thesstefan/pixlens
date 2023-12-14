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
        model_owlvit, self.owlvit_processor = eval_owl_vit.load_owlvit(
            owlvit_type, device=device
        )

        super(OwlVitSam, self).__init__(
            model_owlvit, sam_predictor, detection_confidence_threshold
        )

    def transform_owlvit_output(
        self, owlvit_results: list[dict], prompt: list[str]
    ) -> list:
        results_new = []
        for result in owlvit_results:
            scores = result["scores"]
            labels = [prompt[id] for id in result["labels"].tolist()]
            boxes = result["boxes"]

            detection_output = interfaces.DetectionOutput(
                bounding_boxes=boxes, logits=scores, phrases=labels
            )
            results_new.append(detection_output)
        return results_new

    def detect_with_owlvit(self, prompt: str, image_path: str) -> list:
        image = Image.open(image_path)
        prompts = prompt.split(",")
        inputs = self.owlvit_processor(
            text=prompts, images=image, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.promptable_detection_model(**inputs)

        results = self.owlvit_processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.detection_confidence_threshold,
            target_sizes=torch.Tensor([image.size[::-1]]).to(self.device),
        )
        results = self.transform_owlvit_output(results, prompts)
        return results

    def detect_and_segment(self, prompt: str, image_path: str):
        detection_output = self.detect_with_owlvit(prompt, image_path)[0]
        segmentation_output = self.bbox_segmentation_model.segment(
            detection_output.bounding_boxes, image_path
        )
        return segmentation_output, detection_output
