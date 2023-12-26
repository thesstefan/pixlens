import torch

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.detection.utils import get_detection_segmentation_result_of_target


class ObjectAddition(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        to_attribute = evaluation_input.updated_strings.to_attribute
        is_to_in_edited = (
            1
            if get_detection_segmentation_result_of_target(
                evaluation_input.edited_detection_segmentation_result,
                to_attribute,
            ).detection_output.logits
            else 0
        )
        is_category_in_edited = (
            1
            if get_detection_segmentation_result_of_target(
                evaluation_input.edited_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits
            else 0
        )
        is_category_in_input = bool(
            get_detection_segmentation_result_of_target(
                evaluation_input.input_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits,
        )

        return evaluation_interfaces.EvaluationOutput(
            score=is_category_in_edited * is_to_in_edited,
            success=is_category_in_input,
        )
