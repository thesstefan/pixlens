import torch

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.detection.utils import get_detection_segmentation_result_of_target


class ObjectAddition(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        (
            object_in_input,
            object_not_in_output,
        ) = self.is_object_in_input_and_not_in_output(evaluation_input)
        score_error = evaluation_interfaces.EvaluationOutput(
            score=0, success=False
        )
        if object_in_input and object_not_in_output:
            score = evaluation_interfaces.EvaluationOutput(
                success=True,
                score=1,
            )
        elif not object_in_input:
            score = score_error
        else:
            score = evaluation_interfaces.EvaluationOutput(
                success=True,
                score=0,
            )
        return score

    def is_object_in_input_and_not_in_output(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> tuple[bool, bool]:
        is_category_in_edited = bool(
            get_detection_segmentation_result_of_target(
                evaluation_input.edited_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits,
        )

        is_category_in_input = bool(
            get_detection_segmentation_result_of_target(
                evaluation_input.input_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits,
        )
        return (is_category_in_input, not is_category_in_edited)
