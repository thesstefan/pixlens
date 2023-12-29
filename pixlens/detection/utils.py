import torch

from pixlens.detection import interfaces


def get_detection_segmentation_result_of_target(
    detection_segmentation_result: interfaces.DetectionSegmentationResult,
    target: str,
) -> interfaces.DetectionSegmentationResult:
    detection_output = detection_segmentation_result.detection_output
    segmentation_output = detection_segmentation_result.segmentation_output
    try:
        target_index_detection = detection_output.phrases.index(target)
    except ValueError:
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
    targeted_detection_output = interfaces.DetectionOutput(
        logits=detection_output.logits[target_index_detection],
        bounding_boxes=detection_output.bounding_boxes[target_index_detection],
        phrases=[target],
    )
    targeted_segmentation_output = interfaces.SegmentationOutput(
        logits=segmentation_output.logits[target_index_detection],
        masks=segmentation_output.masks[target_index_detection],
    )

    return interfaces.DetectionSegmentationResult(
        detection_output=targeted_detection_output,
        segmentation_output=targeted_segmentation_output,
    )
