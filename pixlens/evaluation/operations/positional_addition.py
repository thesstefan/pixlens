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


def unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class PositionalAddition(OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationOutput:
        to_attribute = evaluation_input.updated_strings.to_attribute
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
            return self.handle_no_category_in_input()
        if len(category_in_input.detection_output.phrases) > 1:
            logging.warning(
                "More than one object of the same category in the input image.",
            )

        tos_in_edited = self.get_tos_in_edited(
            to_attribute,
            evaluation_input.edited_detection_segmentation_result,
        )
        if len(tos_in_edited.detection_output.phrases) > 1:
            warning_msg = f"More than one '{to_attribute}' in the edited image."
            logging.warning(warning_msg)

        category_center_of_mass = center_of_mass(
            category_in_input.segmentation_output.masks[0],
        )
        to_center_of_mass = center_of_mass(
            tos_in_edited.segmentation_output.masks[0],
        )

        direction_of_movement = self.compute_direction_of_movement(
            category_center_of_mass,
            to_center_of_mass,
        )

        closest_direction = self.compute_closest_direction(
            direction_of_movement,
        )

        if closest_direction == intended_relative_position:
            return self.handle_success()
        return self.handle_failure()

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
            return "below"
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

    def handle_no_category_in_input(self) -> EvaluationOutput:
        return EvaluationOutput(
            edit_specific_score=0,
            success=False,
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
                to_center_of_mass[0] - category_center_of_mass[0],
                to_center_of_mass[1] - category_center_of_mass[1],
            ],
        )

    def compute_closest_direction(
        self,
        direction_of_movement: np.ndarray,
    ) -> str | None:
        direction_vectors = {
            "left": np.array([-1, 0]),
            "right": np.array([1, 0]),
            "top": np.array([0, -1]),
            "below": np.array([0, 1]),
        }

        min_angle = 360.0
        closest_direction = None
        for direction, vector in direction_vectors.items():
            angle = angle_between(direction_of_movement, vector)
            if angle < min_angle:
                min_angle = angle
                closest_direction = direction

        return closest_direction

    def handle_success(self) -> EvaluationOutput:
        return EvaluationOutput(
            edit_specific_score=1,
            success=True,
        )

    def handle_failure(self) -> EvaluationOutput:
        return EvaluationOutput(
            edit_specific_score=0,
            success=True,
        )
