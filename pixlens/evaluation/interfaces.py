import dataclasses
from enum import Enum
from typing import Protocol

from PIL import Image

from pixlens.detection.interfaces import DetectionSegmentationResult


class EditType(Enum):
    SIZE = ("size", "Change the size of {category} to {to}")
    COLOR = ("color", "Change the color of {category} to {to}")
    OBJECT_ADDITION = ("object_addition", "Add a {to} to the image")
    POSITIONAL_ADDITION = (
        "positional_addition",
        "Add a {to} the {category}",
    )
    OBJECT_REMOVAL = ("object_removal", "Remove {category}")
    OBJECT_REPLACEMENT = ("object_replacement", "Replace {from_} with {to}")
    POSITION_REPLACEMENT = ("position_replacement", "Move {from_} to {to}")
    OBJECT_DUPLICATION = ("object_duplication", "Duplicate {category}")
    TEXTURE = ("texture", "Change the texture of {category} to {to}")
    ACTION = ("action", "{category} doing {to}")
    VIEWPOINT = ("viewpoint", "Change the viewpoint to {to}")
    BACKGROUND = ("background", "Change the background to {to}")
    STYLE = ("style", "Change the style of {category} to {to}")
    SHAPE = ("shape", "Change the shape of {category} to {to}")
    ALTER_PARTS = ("alter_parts", "{to} to {category}")

    def __init__(self, type_name: str, prompt: str) -> None:
        self.type_name = type_name
        self.prompt = prompt

    @classmethod
    def from_type_name(cls, type_name: str) -> "EditType":
        for edit_type in cls:
            if edit_type.type_name == type_name:
                return edit_type
        error_msg = f"No such EditType with type_name: {type_name}"
        raise ValueError(error_msg)


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


@dataclasses.dataclass
class EvaluationInput:
    input_image: Image.Image
    edited_image: Image.Image
    prompt: str
    input_detection_segmentation_result: DetectionSegmentationResult
    edited_detection_segmentation_result: DetectionSegmentationResult
    edit: Edit


class OperationEvaluation(Protocol):
    def evaluate_edit(
        self,
        original_detection_segmentation_result: DetectionSegmentationResult,
        edited_detection_segmentation_result: DetectionSegmentationResult,
        image_editing_output: Image.Image,
    ) -> EvaluationOutput:
        ...
