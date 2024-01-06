import torch

from pixlens.detection import interfaces
from pixlens.detection.grounded_sam import GroundedSAM
from pixlens.detection.owl_vit_sam import OwlViTSAM


def get_detection_segmentation_result_of_target(
    detection_segmentation_result: interfaces.DetectionSegmentationResult,
    target: str,
) -> interfaces.DetectionSegmentationResult:
    detection_output = detection_segmentation_result.detection_output
    segmentation_output = detection_segmentation_result.segmentation_output
    if target not in detection_output.phrases:
        return interfaces.DetectionSegmentationResult(
            detection_output=interfaces.DetectionOutput(
                logits=torch.tensor([]),
                bounding_boxes=torch.tensor([]),
                phrases=[],
            ),
            segmentation_output=interfaces.SegmentationOutput(
                logits=torch.tensor([]),
                masks=torch.tensor([]),
            ),
        )

    # get indices of target in detection output by comparing
    # the phrases in detection output with the target
    target_idxs = [
        idx
        for idx, phrase in enumerate(detection_output.phrases)
        if phrase == target
    ]

    targeted_detection_output = interfaces.DetectionOutput(
        logits=detection_output.logits[target_idxs],
        bounding_boxes=detection_output.bounding_boxes[target_idxs],
        phrases=[target],
    )
    targeted_segmentation_output = interfaces.SegmentationOutput(
        logits=segmentation_output.logits[target_idxs],
        masks=segmentation_output.masks[target_idxs],
    )

    return interfaces.DetectionSegmentationResult(
        detection_output=targeted_detection_output,
        segmentation_output=targeted_segmentation_output,
    )


def get_separator(model: interfaces.PromptDetectAndBBoxSegmentModel) -> str:
    if isinstance(model, GroundedSAM):
        return "."
    if isinstance(model, OwlViTSAM):
        return ","
    raise ValueError("Invalid model type")
