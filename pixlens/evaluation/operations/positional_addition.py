import logging

import numpy as np
import torch

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation.interfaces import (
    DetectionSegmentationResult,
    EvaluationInput,
    EvaluationOutput,
    OperationEvaluation,
)
from pixlens.evaluation.multiplicity_resolver import (
    MultiplicityResolution,
    select_one_2d,
)
from pixlens.evaluation.utils import (
    angle_between,
    center_of_mass,
)
from pixlens.visualization.annotation import draw_center_of_masses


class PositionalAddition(OperationEvaluation):
    def __init__(self) -> None:
        self.category_input_resolution = MultiplicityResolution.LARGEST
        self.category_edited_resolution = MultiplicityResolution.LARGEST

    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationOutput:
        # [Note]: to_attribute looks like "X to left of"
        # and to_attribute_for_detection looks like "X"
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
                "More than one object of the same category in the input image."
                " When evaluating a positional addition operation.",
            )

        category_in_edited = self.get_category_in_input(
            evaluation_input.updated_strings.category,
            evaluation_input.edited_detection_segmentation_result,
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

        selected_category_idx_in_input = select_one_2d(
            category_in_input.segmentation_output.masks.cpu().numpy(),
            self.category_input_resolution,
            confidences=category_in_input.detection_output.logits.cpu().numpy(),
            relative_mask=None,
        )
        category_mask_input = category_in_input.segmentation_output.masks[
            selected_category_idx_in_input
        ]

        selected_to_idx_in_edited_idx = select_one_2d(
            tos_in_edited.segmentation_output.masks.cpu().numpy(),
            self.category_edited_resolution,
            confidences=tos_in_edited.detection_output.logits.cpu().numpy(),
        )
        to_mask_edited = tos_in_edited.segmentation_output.masks[
            selected_to_idx_in_edited_idx
        ]

        category_center_of_mass = center_of_mass(
            category_mask_input,
        )
        to_center_of_mass = center_of_mass(
            to_mask_edited,
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
            evaluation_input.input_image.size,
        )

        to_in_edited_score = self.compute_score(
            direction_of_movement,
            intended_relative_position,
        ).edit_specific_score

        category_in_edited_score = float(
            len(category_in_edited.detection_output.bounding_boxes) > 0,
        )

        final_score = (
            ((to_in_edited_score + category_in_edited_score) / 2)
            if to_in_edited_score > 0
            else 0
        )

        return EvaluationOutput(
            edit_specific_score=final_score,
            success=True,
        )

    def get_intended_relative_position(
        self,
        to_attribute: str | None,
    ) -> str | None:
        if to_attribute is None:
            return None

        for direction in ["left", "right", "top", "below"]:
            if direction in to_attribute:
                return direction
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
        input_image_size: tuple[int, int],
        min_required_movement: float = 0.05,
    ) -> np.ndarray:
        direction_of_movement = np.array(
            [
                to_center_of_mass[1] - category_center_of_mass[1],
                category_center_of_mass[0] - to_center_of_mass[0],
            ],
        )
        if np.linalg.norm(
            direction_of_movement,
        ) < min_required_movement * np.linalg.norm(
            np.array(input_image_size),
        ):
            direction_of_movement = np.array([0, 0])
        return direction_of_movement

    def compute_score(
        self,
        direction_of_movement: np.ndarray,
        intended_relative_position: str,
    ) -> EvaluationOutput:
        if np.isclose(np.linalg.norm(direction_of_movement), 0):
            # no movement whatsoever
            return EvaluationOutput(
                edit_specific_score=0,
                success=True,
            )

        direction_vectors = {
            "left": np.array([-1, 0]),
            "right": np.array([1, 0]),
            "top": np.array([0, 1]),
            "below": np.array([0, -1]),
        }

        angle = angle_between(
            direction_of_movement,
            direction_vectors[intended_relative_position],
        )

        angle_in_degrees = float(np.rad2deg(angle))

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
