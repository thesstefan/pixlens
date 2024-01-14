import logging

import numpy as np
import numpy.typing as npt

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.multiplicity_resolver import (
    MultiplicityResolution,
    select_one_2d,
)
from pixlens.evaluation.utils import (
    compute_area_ratio,
    is_small_area_within_big_area,
    pad_into_shape_2d,
)


def size_change_applied_correctly(
    direction: str,
    input_mask: npt.NDArray[np.bool_],
    output_mask: npt.NDArray[np.bool_],
    minimum_required_change: float = 0.1,
) -> bool:
    if direction == "small":
        return compute_area_ratio(
            numerator=output_mask,
            denominator=input_mask,
        ) < (1 - minimum_required_change)
    if direction == "large":
        return compute_area_ratio(
            numerator=output_mask,
            denominator=input_mask,
        ) > (1 + minimum_required_change)
    error_msg = f"Invalid direction: {direction}"
    raise ValueError(error_msg)


class SizeEdit(evaluation_interfaces.OperationEvaluation):
    def __init__(self) -> None:
        self.category_input_resolution: MultiplicityResolution = (
            MultiplicityResolution.LARGEST
        )
        self.category_edited_resolution: MultiplicityResolution = (
            MultiplicityResolution.CLOSEST
        )

    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        category = evaluation_input.updated_strings.category
        category_in_input = get_detection_segmentation_result_of_target(
            evaluation_input.input_detection_segmentation_result,
            category,
        )

        if len(category_in_input.detection_output.bounding_boxes) == 0:
            warning_msg = f"Category {category} not detected in input image"
            " when evaluating size edit."
            logging.warning(warning_msg)
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0.0,
                success=False,
            )

        selected_category_idx_in_input = select_one_2d(
            category_in_input.segmentation_output.masks.cpu().numpy(),
            self.category_input_resolution,
            confidences=category_in_input.detection_output.logits.cpu().numpy(),
            relative_mask=None,
        )
        category_mask_input = np.squeeze(
            (
                category_in_input.segmentation_output.masks[
                    selected_category_idx_in_input
                ]
            )
            .cpu()
            .numpy(),
        )

        category_in_edited = get_detection_segmentation_result_of_target(
            evaluation_input.edited_detection_segmentation_result,
            category,
        )

        if len(category_in_edited.detection_output.bounding_boxes) == 0:
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0.0,
                success=True,
            )

        selected_category_idx_in_edited_idx = select_one_2d(
            category_in_edited.segmentation_output.masks.cpu().numpy(),
            self.category_edited_resolution,
            confidences=category_in_edited.detection_output.logits.cpu().numpy(),
            relative_mask=category_mask_input,
        )
        category_mask_edited = np.squeeze(
            (
                category_in_edited.segmentation_output.masks[
                    selected_category_idx_in_edited_idx
                ]
            )
            .cpu()
            .numpy(),
        )

        # 2 - Check if resize is small or big and compute area difference
        transformation = evaluation_input.edit.to_attribute

        padded_category_mask_edited = pad_into_shape_2d(
            category_mask_edited,
            category_mask_input.shape,
        )

        if size_change_applied_correctly(
            transformation,
            input_mask=category_mask_input,
            output_mask=category_mask_edited,
        ):
            small_movement_score = float(
                is_small_area_within_big_area(
                    input_mask=category_mask_input,
                    edited_mask=padded_category_mask_edited,
                ),
            )
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=(small_movement_score + 1) / 2,
                success=True,
            )
        # therefore if change in size is applied correctly, but the edit moved
        # the object, the score is 0.5
        # if change in size is not applied correctly, the score is 0
        return evaluation_interfaces.EvaluationOutput(
            edit_specific_score=0.0,
            success=True,
        )
