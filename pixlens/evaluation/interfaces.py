import dataclasses
import enum
from typing import Protocol

from pixlens.detection.interfaces import DetectionSegmentationResult
from pixlens.editing.interfaces import ImageEditingOutput


class EditType(enum.StrEnum):
    SIZE = "size"
    COLOR = "color"
    OBJECT_CHANGE = "object_change"
    OBJECT_ADDITION = "object_addition"
    POSITIONAL_ADDITION = "positional_addition"
    OBJECT_REMOVAL = "object_removal"
    OBJECT_REPLACEMENT = "object_replacement"
    POSITION_REPLACEMENT = "position_replacement"
    OBJECT_DUPLICATION = "object_duplication"
    TEXTURE = "texture"
    ACTION = "action"
    VIEWPOINT = "viewpoint"
    BACKGROUND = "background"
    STYLE = "style"
    SHAPE = "shape"
    ALTER_PARTS = "alter_parts"


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
