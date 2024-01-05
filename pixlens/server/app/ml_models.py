import enum

from flask import current_app

from pixlens.detection import load_detect_segment_model_from_yaml
from pixlens.detection.interfaces import PromptDetectAndBBoxSegmentModel
from pixlens.editing import load_editing_model_from_yaml
from pixlens.editing.interfaces import PromptableImageEditingModel


class InferenceType(enum.StrEnum):
    EDITING = "EDITING"
    DETECTION = "DETECTION"


editing_model: PromptableImageEditingModel | None = None
detect_segment_model: PromptDetectAndBBoxSegmentModel | None = None


def load_model() -> None:
    # TODO(thesstefan): Find a better way of doing this
    global editing_model  # noqa: PLW0603
    global detect_segment_model  # noqa: PLW0603

    inference_type = InferenceType(current_app.config["INFERENCE_TYPE"])

    if inference_type == InferenceType.EDITING:
        editing_model = load_editing_model_from_yaml(
            current_app.config["MODEL_PARAMS_YAML"],
        )
    else:
        detect_segment_model = load_detect_segment_model_from_yaml(
            current_app.config["MODEL_PARAMS_YAML"],
        )
