import dataclasses
import enum
from typing import Protocol

from PIL import Image

from pixlens.detection.interfaces import DetectionSegmentationResult


class EditType(enum.StrEnum):
    SIZE = "size"
    COLOR = "color"
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


@dataclasses.dataclass
class Edit:
    edit_id: int
    image_path: str
    image_id: int
    category: str
    edit_type: EditType
    from_attribute: str
    to_attribute: str


class OperationEvaluation(Protocol):
    def evaluate_edit(
        self,
        original_detection_segmentation_result: DetectionSegmentationResult,
        edited_detection_segmentation_result: DetectionSegmentationResult,
        image_editing_output: Image.Image,
    ) -> EvaluationOutput:
        ...
