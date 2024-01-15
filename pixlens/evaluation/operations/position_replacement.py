import logging

import numpy as np
import numpy.typing as npt
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


class PositionReplacement(OperationEvaluation):
    def __init__(self) -> None:
        self.category_input_resolution = MultiplicityResolution.LARGEST
        self.category_edited_resolution = MultiplicityResolution.LARGEST

    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationOutput:
        from_attribute = evaluation_input.edit.from_attribute
        to_attribute = evaluation_input.edit.to_attribute
        initial_position = from_attribute
        intended_position = to_attribute

        if to_attribute is None:
            return self.handle_missing_to_attribute()
        if from_attribute is None:
            return self.handle_missing_from_attribute()

        category_in_input = self.get_category_in_input(
            evaluation_input.updated_strings.category,
            evaluation_input.input_detection_segmentation_result,
        )
        if len(category_in_input.detection_output.bounding_boxes) == 0:
            return self.handle_no_category_in_input(
                evaluation_input.updated_strings.category,
            )
        if len(category_in_input.detection_output.bounding_boxes) > 1:
            logging.warning(
                "More than one object of the same category in the input image."
                " When evaluating a position replacement operation.",
            )

        category_in_edited = self.get_category_in_edited(
            evaluation_input.updated_strings.category,
            evaluation_input.edited_detection_segmentation_result,
        )

        if len(category_in_edited.detection_output.phrases) == 0:
            return self.handle_no_category_in_edited()

        selected_category_idx_in_input = select_one_2d(
            category_in_input.segmentation_output.masks.cpu().numpy(),
            self.category_input_resolution,
            confidences=category_in_input.detection_output.logits.cpu().numpy(),
            relative_mask=None,
        )

        category_mask_input = category_in_input.segmentation_output.masks[
            selected_category_idx_in_input
        ]

        category_pos_initial = center_of_mass(category_mask_input)

        selected_category_idx_in_edited = select_one_2d(
            category_in_edited.segmentation_output.masks.cpu().numpy(),
            self.category_edited_resolution,
            confidences=category_in_edited.detection_output.logits.cpu().numpy(),
            relative_mask=None,
        )
        category_mask_edited = category_in_edited.segmentation_output.masks[
            selected_category_idx_in_edited
        ]

        category_pos_end = center_of_mass(category_mask_edited)

        draw_center_of_masses(
            evaluation_input.annotated_input_image,
            category_pos_initial,
            category_pos_end,
        )

        draw_center_of_masses(
            evaluation_input.annotated_edited_image,
            category_pos_initial,
            category_pos_end,
        )

        direction_of_movement = self.compute_direction_of_movement(
            category_pos_initial,
            category_pos_end,
            evaluation_input.edited_image.size,
        )

        relative_position_change_score = self.compute_relative_score(
            direction_of_movement,
            initial_position=initial_position,
            intended_relative_position=intended_position,
        )

        if relative_position_change_score > 0:
            absolute_position_change_score = self.compute_absolute_score(
                initial_position,
                intended_position,
                category_pos_end,
                image_width=evaluation_input.edited_image.size[0],
            )

            return EvaluationOutput(
                edit_specific_score=max(
                    (
                        relative_position_change_score
                        + absolute_position_change_score
                    )
                    / 2,
                    0,
                ),
                success=True,
            )
        return EvaluationOutput(
            edit_specific_score=0,
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

    def handle_missing_to_attribute(self) -> EvaluationOutput:
        logging.warning(
            "No {to} attribute provided in a position replacement operation.",
        )
        return EvaluationOutput(
            edit_specific_score=0,
            success=False,
        )

    def handle_missing_from_attribute(self) -> EvaluationOutput:
        logging.warning(
            "No {from} attribute provided in a position replacement operation.",
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
        warning_msg = f"No {category} (category) detected in the input image."
        logging.warning(warning_msg)
        return EvaluationOutput(
            edit_specific_score=0,
            success=False,
        )

    def get_category_in_edited(
        self,
        category: str,
        edited_detection_segmentation_result: DetectionSegmentationResult,
    ) -> DetectionSegmentationResult:
        return get_detection_segmentation_result_of_target(
            edited_detection_segmentation_result,
            category,
        )

    def handle_no_category_in_edited(
        self,
    ) -> EvaluationOutput:
        return EvaluationOutput(
            edit_specific_score=0,
            success=True,
        )

    def compute_direction_of_movement(
        self,
        ini: tuple[float, float],
        end: tuple[float, float],
        image_size: tuple[int, int],
        min_required_movement: float = 0.05,
    ) -> npt.NDArray[np.float64]:
        direction = np.array(
            [
                end[1] - ini[1],
                ini[0] - end[0],
            ],
        )
        # if direction vector length is too small in comparison
        # to the size of the input image
        # the direction vector is considered to be 0
        if np.linalg.norm(direction) < min_required_movement * np.linalg.norm(
            image_size,
        ):
            direction = np.array([0, 0])
        return direction

    def compute_relative_score(
        self,
        direction_of_movement: np.ndarray,
        *,
        initial_position: str,
        intended_relative_position: str,
    ) -> float:
        if np.isclose(np.linalg.norm(direction_of_movement), 0):
            # no movement whatsoever
            return 0

        direction_vectors = {
            "left": np.array([-1, 0]),
            "right": np.array([1, 0]),
        }

        if intended_relative_position == "center":
            if initial_position == "left":
                intended_relative_position = "right"
            elif initial_position == "right":
                intended_relative_position = "left"

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

        return max(0, (90 - angle_in_degrees) / 90)

    # function that given the width of the image and a x coordinate
    # returns the position of the coordinate in the image classified
    # as left, center or right
    def get_position_name(
        self,
        x: float,
        image_width: int,
    ) -> str:
        if x < image_width / 3:
            return "left"
        if x > 2 * image_width / 3:
            return "right"
        return "center"

    # function to compute the absolute score
    # divide the image X-axis in 3 parts, left, center and right
    # if the final position is in the intended position, the score is 1
    # if the final position is in the opposite position, the score is -1
    # if the final position is in the same position, the score is 0
    def compute_absolute_score(
        self,
        initial_position: str,
        intended_position: str,
        end: tuple[float, float],
        image_width: int,
    ) -> float:
        real_final_position = self.get_position_name(end[1], image_width)

        if initial_position == real_final_position:  # same position
            return 0
        if (
            (  # partially correct position = either too much or too little
                initial_position == "left"
                and intended_position == "right"
                and real_final_position == "center"
            )
            or (
                initial_position == "right"
                and intended_position == "left"
                and real_final_position == "center"
            )
            or (
                initial_position == "left"
                and intended_position == "center"
                and real_final_position == "right"
            )
            or (
                initial_position == "right"
                and intended_position == "center"
                and real_final_position == "left"
            )
        ):
            return 0.5
        if intended_position == real_final_position:  # correct position
            return 1
        return 0  # absolute position change is in the opposite direction
