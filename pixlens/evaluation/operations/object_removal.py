import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.operations.utils import apply_mask


class ObjectRemoval(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        (
            object_in_input,
            object_not_in_output,
        ) = self.is_object_in_input_and_not_in_output(evaluation_input)

        score_error = evaluation_interfaces.EvaluationOutput(
            score=0,
            success=False,
        )
        if object_in_input and object_not_in_output:
            score = evaluation_interfaces.EvaluationOutput(
                success=True,
                score=1,
            )
            ssim = self.calculate_ssim(evaluation_input)
            score.score = score.score * ssim
            return score
        if not object_in_input:
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
            ).detection_output.logits.any(),
        )

        is_category_in_input = bool(
            get_detection_segmentation_result_of_target(
                evaluation_input.input_detection_segmentation_result,
                evaluation_input.updated_strings.category,
            ).detection_output.logits.any(),
        )
        return (is_category_in_input, not is_category_in_edited)

    def calculate_ssim(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> float:
        input_image_np = np.array(evaluation_input.input_image)
        edited_image_np = np.array(evaluation_input.edited_image)
        if edited_image_np.shape != input_image_np.shape:
            edited_image_resized = evaluation_input.edited_image.resize(
                evaluation_input.input_image.size,
                Image.Resampling.LANCZOS,
            )
            edited_image_np = np.array(edited_image_resized)
        ssim_full_image = np.clip(
            ssim(input_image_np, edited_image_np, channel_axis=2),
            0,
            1,
        ).__float__()
        mask = (
            evaluation_input.input_detection_segmentation_result.segmentation_output.masks[
                0
            ]
            .cpu()
            .numpy()
        )
        input_image_masked = apply_mask(input_image_np, mask)
        edited_image_masked = apply_mask(edited_image_np, mask)
        ssim_masked = np.clip(
            ssim(input_image_masked, edited_image_masked, channel_axis=2),
            0,
            1,
        ).__float__()
        return (ssim_full_image + ssim_masked) / 2
