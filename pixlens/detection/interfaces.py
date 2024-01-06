import dataclasses
import logging
from typing import Protocol

import torch
from PIL import Image
from torchvision.ops import nms

from pixlens.base_model import BaseModel


@dataclasses.dataclass
class DetectionOutput:
    bounding_boxes: torch.Tensor
    logits: torch.Tensor
    phrases: list[str]


@dataclasses.dataclass
class SegmentationOutput:
    masks: torch.Tensor
    logits: torch.Tensor


class PromptableDetectionModel(Protocol):
    def detect(
        self,
        prompt: str,
        image: Image.Image,
    ) -> DetectionOutput:
        ...


@dataclasses.dataclass
class DetectionSegmentationResult:
    detection_output: DetectionOutput
    segmentation_output: SegmentationOutput


class BBoxSegmentationModel(Protocol):
    def segment(
        self,
        bbox: torch.Tensor,
        image: Image.Image,
    ) -> SegmentationOutput:
        ...


class PromptableSegmentationModel(Protocol):
    def segment(self, prompt: str, image: Image.Image) -> SegmentationOutput:
        ...


class PromptDetectAndBBoxSegmentModel(BaseModel, PromptableSegmentationModel):
    def __init__(
        self,
        promptable_detection_model: PromptableDetectionModel,
        bbox_segmentation_model: BBoxSegmentationModel,
        detection_confidence_threshold: float,
    ) -> None:
        self.promptable_detection_model = promptable_detection_model
        self.bbox_segmentation_model = bbox_segmentation_model
        self.detection_confidence_threshold = detection_confidence_threshold

    def filter_detection_output(
        self,
        detection_output: DetectionOutput,
    ) -> DetectionOutput:
        confident_predictions = torch.where(
            detection_output.logits > self.detection_confidence_threshold,
        )[0]
        filtered_boxes = detection_output.bounding_boxes[confident_predictions]
        filtered_logits = detection_output.logits[confident_predictions]

        # Apply Non-Maximum Suppression
        iou_threshold = 0.5  # You can adjust this threshold
        keep = nms(filtered_boxes, filtered_logits, iou_threshold)

        # Update detection_output with NMS results
        detection_output.bounding_boxes = filtered_boxes[keep]
        detection_output.logits = filtered_logits[keep]
        detection_output.phrases = [
            detection_output.phrases[i] for i in confident_predictions[keep]
        ]

        return detection_output

    def detect_and_segment(
        self,
        prompt: str,
        image: Image.Image,
    ) -> tuple[SegmentationOutput, DetectionOutput]:
        detection_output = self.promptable_detection_model.detect(
            prompt,
            image,
        )
        detection_output = self.filter_detection_output(detection_output)

        if detection_output.bounding_boxes.shape[0] == 0:
            logging.warning("No objects detected")
            return SegmentationOutput(
                # no objects detected so logits should be empty and same for masks
                masks=torch.tensor([]),
                logits=torch.tensor([]),
            ), detection_output

        segmentation_output = self.bbox_segmentation_model.segment(
            detection_output.bounding_boxes,
            image,
        )

        return segmentation_output, detection_output

    def segment(
        self,
        prompt: str,
        image: Image.Image,
    ) -> SegmentationOutput:
        return self.detect_and_segment(prompt, image)[0]
