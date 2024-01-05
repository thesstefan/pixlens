import logging

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.utils import color_change_applied_correctly


class ColorEdit(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        input_segmentation = evaluation_input.input_detection_segmentation_result.segmentation_output
        edit_segmentation = evaluation_input.edited_detection_segmentation_result.segmentation_output
        if not input_segmentation.masks.any():
            logging.warning(
                "Size edit could not be evaluated, because no object was "
                "present at input",
            )
            return evaluation_interfaces.EvaluationOutput(
                score=-1.0,
                success=False,
            )  # Object wasn't even present at input
        if edit_segmentation.masks.any().item():
            target_color = evaluation_input.edit.to_attribute
            idmax = edit_segmentation.logits.argmax()
            mask_edited = edit_segmentation.masks[idmax]

            if color_change_applied_correctly(
                image=evaluation_input.edited_image,
                mask=mask_edited,
                target_color=target_color,
            ):
                return evaluation_interfaces.EvaluationOutput(
                    score=1.0,
                    success=True,
                )
        return evaluation_interfaces.EvaluationOutput(
            score=0.0,
            success=True,
        )  # Object wasn't present at output or color was not changed correctly
