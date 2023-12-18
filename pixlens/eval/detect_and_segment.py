import torch

from pixlens.utils import interfaces


class PromptDetectAndBBoxSegmentModel(interfaces.PromptableSegmentationModel):
    def __init__(
        self,
        promptable_detection_model: interfaces.PromptableDetectionModel,
        bbox_segmentation_model: interfaces.BBoxSegmentationModel,
        detection_confidence_threshold: float,
    ) -> None:
        self.promptable_detection_model = promptable_detection_model
        self.bbox_segmentation_model = bbox_segmentation_model
        self.detection_confidence_threshold = detection_confidence_threshold

    def filter_detection_output(
        self, detection_output: interfaces.DetectionOutput
    ) -> interfaces.DetectionOutput:
        confident_predictions = torch.where(
            detection_output.logits > self.detection_confidence_threshold
        )[0]

        detection_output.logits = detection_output.logits[confident_predictions]
        detection_output.bounding_boxes = detection_output.bounding_boxes[
            confident_predictions
        ]

        detection_output.phrases = [
            phrase
            for index, phrase in enumerate(detection_output.phrases)
            if index in confident_predictions
        ]

        return detection_output

    def detect_and_segment(
        self, prompt: str, image_path: str
    ) -> tuple[interfaces.SegmentationOutput, interfaces.DetectionOutput]:
        detection_output = self.promptable_detection_model.detect(prompt, image_path)
        detection_output = self.filter_detection_output(detection_output)

        segmentation_output = self.bbox_segmentation_model.segment(
            detection_output.bounding_boxes, image_path
        )

        return segmentation_output, detection_output

    def segment(self, prompt: str, image_path: str) -> interfaces.SegmentationOutput:
        return self.detect_and_segment(prompt, image_path)[0]
