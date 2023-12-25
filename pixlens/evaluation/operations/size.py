from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.utils import (
    compute_area_ratio,
    is_small_area_within_big_area,
)


class SizeEdit(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        # 1 - Check if object is present in both input and output:
        input_segmentation = evaluation_input.input_detection_segmentation_result.segmentation_output
        edit_segmentation = evaluation_input.edited_detection_segmentation_result.segmentation_output
        if evaluation_input.edited_detection_segmentation_result.detection_output.phrases:
            # 2 - Check if resize is small or big and compute area difference
            transformation = evaluation_input.edit.to_attribute
            mask_input = input_segmentation.masks[0]
            idmax = edit_segmentation.logits.argmax()
            mask_edited = edit_segmentation.masks[idmax]

            if (transformation == "small") and compute_area_ratio(
                numerator=mask_input,
                denominator=mask_edited,
            ) > 1:
                return evaluation_interfaces.EvaluationOutput(
                    score=float(
                        is_small_area_within_big_area(
                            small_area=mask_edited,
                            big_area=mask_input,
                        ),
                    ),
                )
            if (transformation == "big") and compute_area_ratio(
                numerator=mask_input,
                denominator=mask_edited,
            ) < 1:
                return evaluation_interfaces.EvaluationOutput(
                    score=float(
                        is_small_area_within_big_area(
                            small_area=mask_input,
                            big_area=mask_edited,
                        ),
                    ),
                )
        return evaluation_interfaces.EvaluationOutput(
            score=0.0,
        )  # Object wasn't present at output or area was indeed bigger / smaller
