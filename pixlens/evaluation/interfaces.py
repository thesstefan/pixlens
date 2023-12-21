import dataclasses
from typing import Protocol

from pixlens.detection.interfaces import DetectionSegmentationResult
from pixlens.editing.interfaces import ImageEditingOutput


@dataclasses.dataclass
class EvaluationOutput:
    score: float


class OperationEvaluation(Protocol):
    def evaluate_edit(
        self,
        original_detection_segmentation_result: DetectionSegmentationResult,
        edited_detection_segmentation_result: DetectionSegmentationResult,
        image_editing_output: ImageEditingOutput,
    ) -> EvaluationOutput:
        ...
