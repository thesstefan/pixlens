import dataclasses
import enum
import json
import pathlib
from typing import Protocol

from PIL import Image

from pixlens.detection.interfaces import DetectionSegmentationResult


class EditType(enum.StrEnum):
    SIZE = "size"
    COLOR = "color"
    OBJECT_ADDITION = "object_addition"
    POSITIONAL_ADDITION = "positional_addition"
    OBJECT_REMOVAL = "object_removal"
    SINGLE_INSTANCE_REMOVAL = "single_instance_removal"
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
        edit_specific_summary = {
            "success": self.success,
            "edit_specific_score": self.edit_specific_score,
        }

        json_str = json.dumps(edit_specific_summary, indent=4)
        score_json_path = save_dir / "edit_specific_scores.json"

        # guarantee that the parent directory exists
        score_json_path.parent.mkdir(parents=True, exist_ok=True)

        with score_json_path.open("w") as score_json:
            score_json.write(json_str)

        if self.artifacts:
            self.artifacts.persist(save_dir)


@dataclasses.dataclass
class Edit:
    edit_id: int
    image_path: str
    image_id: str
    category: str
    edit_type: EditType
    from_attribute: str | None
    to_attribute: str | None
    instruction_prompt: str
    description_prompt: str


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
