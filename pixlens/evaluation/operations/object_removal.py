from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.utils import compute_ssim


class ObjectRemoval(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        (
            object_in_input,
            object_not_in_output,
        ) = self.is_object_in_input_and_not_in_output(evaluation_input)

        evaluation_output_error = evaluation_interfaces.EvaluationOutput(
            success=False,
            edit_specific_score=0,
            ssim_score=None,
        )
        if object_in_input and object_not_in_output:
            return evaluation_interfaces.EvaluationOutput(
                success=True,
                edit_specific_score=1,
                ssim_score=compute_ssim(evaluation_input),
            )
        if not object_in_input:
            return evaluation_output_error

        return evaluation_interfaces.EvaluationOutput(
            success=True,
            edit_specific_score=0,
            ssim_score=compute_ssim(evaluation_input),
        )

    def is_object_in_input_and_not_in_output(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> tuple[bool, bool]:
        is_category_in_edited = bool(
            get_detection_segmentation_result_of_target(
                evaluation_input.edited_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits.any(),
        )

        is_category_in_input = bool(
            get_detection_segmentation_result_of_target(
                evaluation_input.input_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits.any(),
        )
        return (is_category_in_input, not is_category_in_edited)
