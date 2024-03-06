import logging

import numpy as np
from torchvision.ops import box_iou

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.multiplicity_resolver import (
    MultiplicityResolution,
    select_one_2d,
)


class ObjectReplacement(evaluation_interfaces.OperationEvaluation):
    def __init__(self) -> None:
        self.from_input_resolution: MultiplicityResolution = (
            MultiplicityResolution.LARGEST
        )
        self.to_edited_resolution: MultiplicityResolution = (
            MultiplicityResolution.CLOSEST
        )

    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        from_attribute = evaluation_input.updated_strings.from_attribute
        to_attribute = evaluation_input.updated_strings.to_attribute
        if to_attribute is None:
            logging.warning(
                "No {to} attribute provided in an "
                "object replacement operation.",
            )
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0,
                success=False,
            )
        if from_attribute is None:
            logging.warning(
                "No {from} attribute provided in an "
                "object replacement operation.",
            )
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0,
                success=False,
            )

        froms_in_input = get_detection_segmentation_result_of_target(
            evaluation_input.input_detection_segmentation_result,
            from_attribute,
        )
        tos_in_edited = get_detection_segmentation_result_of_target(
            evaluation_input.edited_detection_segmentation_result,
            to_attribute,
        )

        if len(froms_in_input.detection_output.bounding_boxes) == 0:
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0,
                success=False,
            )

        if len(tos_in_edited.detection_output.bounding_boxes) == 0:
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0,
                success=True,
            )

        selected_from_idx_in_input = select_one_2d(
            froms_in_input.segmentation_output.masks.cpu().numpy(),
            self.from_input_resolution,
        )
        from_mask_input = np.squeeze(
            (
                froms_in_input.segmentation_output.masks[
                    selected_from_idx_in_input
                ]
            )
            .cpu()
            .numpy(),
        )

        selected_to_idx_in_edited = select_one_2d(
            tos_in_edited.segmentation_output.masks.cpu().numpy(),
            self.to_edited_resolution,
            relative_mask=from_mask_input,
        )
        to_mask_edited = np.squeeze(
            (tos_in_edited.segmentation_output.masks[selected_to_idx_in_edited])
            .cpu()
            .numpy(),
        )

        # if at least minimal intersection between the from and to masks
        # then the operation is successful
        intersection = np.logical_and(from_mask_input, to_mask_edited)
        if np.count_nonzero(intersection) > 0:
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=1,
                success=True,
            )
        return evaluation_interfaces.EvaluationOutput(
            edit_specific_score=0,
            success=True,
        )
