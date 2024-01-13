import torch

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces


class ObjectAddition(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        to_attribute = evaluation_input.updated_strings.to_attribute
        if to_attribute is None:
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0,
                success=False,
            )
        is_to_in_edited = (
            1
            if get_detection_segmentation_result_of_target(
                evaluation_input.edited_detection_segmentation_result,
                to_attribute,
            ).detection_output.logits.size()
            != torch.Size([0])
            else 0
        )
        is_category_in_edited = (
            1
            if get_detection_segmentation_result_of_target(
                evaluation_input.edited_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits.size()
            != torch.Size([0])
            else 0
        )
        is_category_in_input = bool(
            get_detection_segmentation_result_of_target(
                evaluation_input.input_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits.size()
            != torch.Size([0]),
        )
        score = is_to_in_edited * (is_to_in_edited + is_category_in_edited) / 2
        return evaluation_interfaces.EvaluationOutput(
            edit_specific_score=score,
            success=is_category_in_input,
        )
