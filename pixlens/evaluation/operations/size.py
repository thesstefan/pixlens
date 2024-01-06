import torch

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.utils import (
    compute_area_ratio,
    is_small_area_within_big_area,
)


def size_change_applied_correctly(
    direction: str,
    input_mask: torch.Tensor,
    output_mask: torch.Tensor,
) -> bool:
    if direction == "small":
        return (
            compute_area_ratio(
                numerator=output_mask,
                denominator=input_mask,
            )
            < 1
        )
    if direction == "big":
        return (
            compute_area_ratio(
                numerator=output_mask,
                denominator=input_mask,
            )
            > 1
        )
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
            mask_input = input_segmentation.masks[0]
            idmax = edit_segmentation.logits.argmax()
            mask_edited = edit_segmentation.masks[idmax]

            if size_change_applied_correctly(
                transformation,
                mask_input,
                mask_edited,
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
