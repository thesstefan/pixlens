import numpy as np
import torch

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.utils import (
    center_of_mass,
    compute_area_ratio,
    is_small_area_within_big_area,
)


def size_change_applied_correctly(
    direction: str,
    input_mask: torch.Tensor,
    output_mask: torch.Tensor,
    minimum_required_change: float = 0.1,
) -> bool:
    if direction == "small":
        return compute_area_ratio(
            numerator=output_mask,
            denominator=input_mask,
        ) < (1 - minimum_required_change)
    if direction == "big":
        return compute_area_ratio(
            numerator=output_mask,
            denominator=input_mask,
        ) > (1 + minimum_required_change)
    error_msg = f"Invalid direction: {direction}"
    raise ValueError(error_msg)


class SizeEdit(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        # 1 - Check if object is present in both input and output:
        input_segmentation = evaluation_input.input_detection_segmentation_result.segmentation_output
        edit_segmentation = evaluation_input.edited_detection_segmentation_result.segmentation_output
        if not input_segmentation.masks.any():
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=-1.0,
                success=False,
            )  # Object wasn't even present at input
        if edit_segmentation.masks.any():
            # Code continues here...
            # 2 - Check if resize is small or big and compute area difference
            transformation = evaluation_input.edit.to_attribute

            # get center of mass of the category in the input image
            mask_input = input_segmentation.masks[0]
            category_center_of_mass = center_of_mass(
                mask_input,
            )

            # retrieve index of the object in edited image that is closer to the input object
            # distance is computed as euclidean distance between the center of mass of the
            # category in the input image and the center of mass of the object in the edited image
            edited_centers_of_mass = [
                center_of_mass(mask) for mask in edit_segmentation.masks
            ]

            closest_obj_index = int(
                np.argmin(
                    [
                        np.linalg.norm(
                            np.array(category_center_of_mass)
                            - np.array(edited_center_of_mass),
                        )
                        for edited_center_of_mass in edited_centers_of_mass
                    ],
                ),
            )

            mask_edited = edit_segmentation.masks[closest_obj_index]

            if size_change_applied_correctly(
                transformation,
                input_mask=mask_input,
                output_mask=mask_edited,
            ):
                return evaluation_interfaces.EvaluationOutput(
                    edit_specific_score=float(
                        is_small_area_within_big_area(
                            input_mask=mask_input,
                            edited_mask=mask_edited,
                        ),
                    ),
                    success=True,
                )
        return evaluation_interfaces.EvaluationOutput(
            edit_specific_score=0.0,
            success=True,
        )  # Object wasn't present at output or area was indeed bigger / smaller
