import logging

import numpy as np

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
from pixlens.evaluation.utils import compute_mask_intersection


class AlterParts(OperationEvaluation):
    def __init__(self) -> None:
        self.to_edited_resolution: MultiplicityResolution = (
            MultiplicityResolution.CLOSEST
        )

    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationOutput:
        to_attribute = evaluation_input.updated_strings.to_attribute
        category = evaluation_input.updated_strings.category

        if to_attribute is None:
            return self.handle_missing_to_attribute()

        category_in_input = self.get_category_in_input(
            category,
            evaluation_input.input_detection_segmentation_result,
        )

        if len(category_in_input.detection_output.bounding_boxes) == 0:
            return self.handle_no_category_in_input(
                category,
            )

        if len(category_in_input.detection_output.bounding_boxes) > 1:
            logging.warning(
                "More than one object of the same category in the input image,"
                " when evaluating an alter parts operation.",
            )

        tos_in_edited = self.get_tos_in_edited(
            to_attribute,
            evaluation_input.edited_detection_segmentation_result,
        )

        if len(tos_in_edited.detection_output.bounding_boxes) == 0:
            return self.handle_no_to_attribute_in_edited(to_attribute)

        # compute score with category in input into account
        score = self.compute_score(
            tos_in_edited,
            category_in_input,
        )

        return EvaluationOutput(
            edit_specific_score=score,
            success=True,
        )

    def handle_missing_to_attribute(self) -> EvaluationOutput:
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

    def get_category_in_edited(
        self,
        category: str,
        edited_detection_segmentation_result: DetectionSegmentationResult,
    ) -> DetectionSegmentationResult:
        return get_detection_segmentation_result_of_target(
            edited_detection_segmentation_result,
            category,
        )

    def handle_no_category_in_input(
        self,
        category: str,
    ) -> EvaluationOutput:
        warning_msg = f"No {category} (category) detected in the input image,"
        " when evaluating an alter parts operation."
        logging.warning(warning_msg)
        return EvaluationOutput(
            edit_specific_score=0,
            success=False,
        )

    def handle_no_category_in_edited(
        self,
        category: str,
    ) -> EvaluationOutput:
        warning_msg = f"No {category} (categiry) detected in the edited image,"
        " when evaluating an alter parts operation."
        logging.warning(warning_msg)
        return EvaluationOutput(
            edit_specific_score=0,
            success=True,
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

    def compute_score(
        self,
        tos_in_edited: DetectionSegmentationResult,
        category: DetectionSegmentationResult,
    ) -> float:
        intersection_ratios = []

        for category_mask in category.segmentation_output.masks:
            closest_to_object_in_edited_idx = select_one_2d(
                tos_in_edited.segmentation_output.masks.cpu().numpy(),
                self.to_edited_resolution,
                relative_mask=np.squeeze(category_mask).cpu().numpy(),
            )

            intersection_ratio = compute_mask_intersection(
                whole=category_mask,
                part=tos_in_edited.segmentation_output.masks[
                    closest_to_object_in_edited_idx
                ],
            )
            intersection_ratios.append(
                intersection_ratio > 0.0,
            )  # 1 if True, 0 if False

        return float(np.mean(intersection_ratios))
