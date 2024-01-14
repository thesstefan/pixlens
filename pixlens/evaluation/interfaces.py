import dataclasses
import enum
from typing import Protocol
import pathlib

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
    BACKGROUND = "background"
    VIEWPOINT = "viewpoint"
    STYLE = "style"
    SHAPE = "shape"
    ALTER_PARTS = "alter_parts"


class Persistable(Protocol):
    def persist(self, save_dir: pathlib.Path) -> None:
        ...


class EvaluationArtifacts(Persistable):
    ...


@dataclasses.dataclass(kw_only=True)
class EvaluationOutput(Persistable):
    success: bool
    edit_specific_score: float
    artifacts: EvaluationArtifacts | None = None

    def persist(self, save_dir: pathlib.Path) -> None:
        if self.artifacts:
            self.artifacts.persist(save_dir)


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
class UpdatedStrings:
    category: str
    from_attribute: str | None
    to_attribute: str | None


@dataclasses.dataclass
class EvaluationInput:
    input_image: Image.Image
    annotated_input_image: Image.Image
    edited_image: Image.Image
    annotated_edited_image: Image.Image
    prompt: str
    input_detection_segmentation_result: DetectionSegmentationResult
    edited_detection_segmentation_result: DetectionSegmentationResult
    edit: Edit
    updated_strings: UpdatedStrings


class OperationEvaluation(Protocol):
    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationOutput:
        ...
