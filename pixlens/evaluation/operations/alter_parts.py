import logging

import numpy as np

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation.interfaces import (
    DetectionSegmentationResult,
    EvaluationInput,
    EvaluationOutput,
    OperationEvaluation,
)
from pixlens.evaluation.utils import compute_mask_intersection


class AlterParts(OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationOutput:
        to_attribute = evaluation_input.updated_strings.to_attribute
        category = evaluation_input.updated_strings.category

        if to_attribute is None:
            return self.handle_no_to_attribute()

        category_in_input = self.get_category_in_input(
            category,
            evaluation_input.input_detection_segmentation_result,
        )

        if len(category_in_input.detection_output.phrases) == 0:
            return self.handle_no_category_in_input(
                category,
            )
        if len(category_in_input.detection_output.phrases) > 1:
            logging.warning(
                "More than one object of the same category in the input image,"
                " when evaluating an alter parts operation.",
            )

        tos_in_edited = self.get_tos_in_edited(
            to_attribute,
            evaluation_input.edited_detection_segmentation_result,
        )
        if len(tos_in_edited.detection_output.phrases) == 0:
            return self.handle_no_to_attribute_in_edited(to_attribute)

        intersection_ratios = []
        for to_mask in tos_in_edited.segmentation_output.masks:
            intersection_ratio = compute_mask_intersection(
                whole=category_in_input.segmentation_output.masks[0],
                # we are assuming that there is only one object for
                # {category} in the input image
                part=to_mask,
            )
            intersection_ratios.append(intersection_ratio)

        average_intersection_ratio = float(np.mean(intersection_ratios))

        return EvaluationOutput(
            edit_specific_score=average_intersection_ratio,
            success=True,
        )

    def handle_no_to_attribute(self) -> EvaluationOutput:
        logging.warning(
            "No {to} attribute provided in an alter parts operation.",
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
        warning_msg = f"No {category} (categiry) detected in the input image,"
        " when evaluating an alter parts operation."
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
            f"No {to_attribute} (to_attribute) detected in the edited image, "
            "when evaluating an alter parts operation."
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
