import logging

import numpy as np

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation.interfaces import (
    DetectionSegmentationResult,
    EvaluationInput,
    EvaluationOutput,
    OperationEvaluation,
)
from pixlens.evaluation.utils import center_of_mass
from pixlens.visualization.annotation import draw_center_of_masses


def unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def radians_to_degrees(radians: float) -> float:
    return radians * 180 / np.pi


class PositionalAddition(OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationOutput:
        to_attribute = evaluation_input.edit.to_attribute
        to_attribute_for_detection = (
            evaluation_input.updated_strings.to_attribute
        )
        intended_relative_position = self.get_intended_relative_position(
            to_attribute,
        )
        if to_attribute is None or intended_relative_position is None:
            return self.handle_no_to_attribute()

        category_in_input = self.get_category_in_input(
            evaluation_input.updated_strings.category,
            evaluation_input.input_detection_segmentation_result,
        )
        if len(category_in_input.detection_output.phrases) == 0:
            return self.handle_no_category_in_input(
                evaluation_input.updated_strings.category,
            )
        if len(category_in_input.detection_output.phrases) > 1:
            logging.warning(
                "More than one object of the same category in the input image.",
            )

        tos_in_edited = self.get_tos_in_edited(
            to_attribute_for_detection
            if to_attribute_for_detection is not None
            else "",
            evaluation_input.edited_detection_segmentation_result,
        )
        if len(tos_in_edited.detection_output.phrases) == 0:
            return self.handle_no_to_attribute_in_edited(
                to_attribute_for_detection
                if to_attribute_for_detection is not None
                else "",
            )

        if len(tos_in_edited.detection_output.phrases) > 1:
            warning_msg = f"More than one '{to_attribute}' in the edited image."
            logging.warning(warning_msg)

        # FIXME(julencosta): we are assuming that there is only one object  # noqa: TD001, TD003, FIX001, E501
        # detected in the edited image and that it is the object that was
        # added. However, this is not always the case.
        category_center_of_mass = center_of_mass(
            category_in_input.segmentation_output.masks[0],
        )
        to_center_of_mass = center_of_mass(
            tos_in_edited.segmentation_output.masks[0],
        )

        draw_center_of_masses(
            evaluation_input.annotated_input_image,
            category_center_of_mass,
            to_center_of_mass,
        )

        draw_center_of_masses(
            evaluation_input.annotated_edited_image,
            category_center_of_mass,
            to_center_of_mass,
        )

        direction_of_movement = self.compute_direction_of_movement(
            category_center_of_mass,
            to_center_of_mass,
        )

        return self.compute_score(
            direction_of_movement,
            intended_relative_position,
        )

    def get_intended_relative_position(
        self,
        to_attribute: str | None,
    ) -> str | None:
        if to_attribute is None:
            return None
        if "left" in to_attribute:
            return "left"
        if "right" in to_attribute:
            return "right"
        if "top" in to_attribute:
            return "top"
        if "below" in to_attribute:
            return "bottom"
        return None

    def handle_no_to_attribute(self) -> EvaluationOutput:
        logging.warning(
            "No {to} attribute provided in a positional addition operation.",
        )
        return EvaluationOutput(
            edit_specific_score=0,
            success=False,
        )

    def get_category_in_input(
        self,
        category: str,
        input_detection_segmentation_result: DetectionSegmentationResult,
    ) -> DetectionSegmentationResult:
        return get_detection_segmentation_result_of_target(
            input_detection_segmentation_result,
            category,
        )

    def handle_no_category_in_input(
        self,
        category: str,
    ) -> EvaluationOutput:
        warning_msg = f"No {category} (categiry) detected in the input image."
        logging.warning(warning_msg)
        return EvaluationOutput(
            edit_specific_score=0,
            success=False,
        )

    def handle_no_to_attribute_in_edited(
        self,
        to_attribute: str,
    ) -> EvaluationOutput:
        warning_msg = (
            f"No {to_attribute} (to_attribute) detected in the edited image."
        )
        logging.warning(warning_msg)
        return EvaluationOutput(
            edit_specific_score=0,
            success=True,
        )

    def get_tos_in_edited(
        self,
        to_attribute: str,
        edited_detection_segmentation_result: DetectionSegmentationResult,
    ) -> DetectionSegmentationResult:
        return get_detection_segmentation_result_of_target(
            edited_detection_segmentation_result,
            to_attribute,
        )

    def compute_direction_of_movement(
        self,
        category_center_of_mass: tuple[float, float],
        to_center_of_mass: tuple[float, float],
    ) -> np.ndarray:
        return np.array(
            [
                to_center_of_mass[1] - category_center_of_mass[1],
                category_center_of_mass[0] - to_center_of_mass[0],
            ],
        )

    def compute_score(
        self,
        direction_of_movement: np.ndarray,
        intended_relative_position: str,
    ) -> EvaluationOutput:
        direction_vectors = {
            "left": np.array([-1, 0]),
            "right": np.array([1, 0]),
            "top": np.array([0, 1]),
            "bottom": np.array([0, -1]),
        }

        angle = angle_between(
            direction_of_movement,
            direction_vectors[intended_relative_position],
        )

        angle_in_degrees = radians_to_degrees(angle)

        # score of edit is a linear interpolation between 0 (perfect angle)
        # and 90 (worst angle), if angle is higher than 90, the score is 0
        # if angle is 0, the score is 1
        if angle_in_degrees < 0:
            logging.warning("Angle is negative")

        score = max(0, (90 - angle_in_degrees) / 90)
        return EvaluationOutput(
            edit_specific_score=score,
            success=True,
        )
