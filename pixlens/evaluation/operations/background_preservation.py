import numpy as np
import torch
from numpy.typing import NDArray

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation import utils as image_utils


class BackgroundPreservation(evaluation_interfaces.GeneralEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
        precomputed_evaluation_output: evaluation_interfaces.EvaluationOutput,
    ) -> None:
        input_image = evaluation_input.input_image
        edited_image = evaluation_input.edited_image
        masks = self.get_masks(evaluation_input)
        masks = [self.mask_into_np(mask) for mask in masks]
        target_shape = masks[0].shape
        reshaped_masks = []
        for mask in masks:
            if mask.shape != masks[0].shape:
                new_mask = np.zeros(target_shape, dtype=mask.dtype)
                new_mask[
                    : min(mask.shape[0], target_shape[0]),
                    : min(mask.shape[1], target_shape[1]),
                ] = mask[
                    : min(mask.shape[0], target_shape[0]),
                    : min(mask.shape[1], target_shape[1]),
                ]
                reshaped_masks.append(new_mask)
            else:
                reshaped_masks.append(mask)

        union_mask = image_utils.compute_union_segmentation_masks(
            reshaped_masks,
        )
        background_preservation_mse = image_utils.extract_decimal_part(
            1
            - image_utils.compute_mse_over_mask(
                input_image,
                edited_image,
                union_mask,
                union_mask,
                background=True,
            )
        )
        background_preservation_ssim = image_utils.compute_ssim_over_mask(
            input_image,
            edited_image,
            union_mask,
            union_mask,
            background=True,
        ) * (1 - union_mask.sum() / union_mask.size)  # TO MENTION IN THE PAPER

        precomputed_evaluation_output.background_preservation_score_mse = (
            background_preservation_mse
        )
        precomputed_evaluation_output.background_preservation_score_ssim = (
            background_preservation_ssim
        )

    def get_masks(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> list[torch.Tensor]:
        edit_type = evaluation_input.edit.edit_type
        edit_type_class = evaluation_interfaces.EditType
        add_type = [
            edit_type_class.OBJECT_ADDITION,
            edit_type_class.POSITIONAL_ADDITION,
        ]
        only_category_type = [
            edit_type_class.TEXTURE,
            edit_type_class.COLOR,
            edit_type_class.SIZE,
            edit_type_class.SHAPE,
            edit_type_class.STYLE,
            edit_type_class.POSITION_REPLACEMENT,
            edit_type_class.VIEWPOINT,
        ]
        masks = [torch.zeros(evaluation_input.input_image.size)]
        if edit_type in add_type:
            indices = image_utils.find_word_indices(
                evaluation_input.edited_detection_segmentation_result.detection_output.phrases,
                evaluation_input.updated_strings.to_attribute,
            )
            masks += [
                evaluation_input.edited_detection_segmentation_result.segmentation_output.masks[
                    index
                ][0]
                for index in indices
            ]
        elif edit_type in only_category_type:
            indices = image_utils.find_word_indices(
                evaluation_input.input_detection_segmentation_result.detection_output.phrases,
                evaluation_input.updated_strings.category,
            )
            masks += [
                evaluation_input.input_detection_segmentation_result.segmentation_output.masks[
                    index
                ][0]
                for index in indices
            ]
        else:
            n = evaluation_input.input_detection_segmentation_result.segmentation_output.masks.size()[  # noqa: E501
                0
            ]
            m = evaluation_input.edited_detection_segmentation_result.segmentation_output.masks.size()[  # noqa: E501
                0
            ]
            masks += [
                evaluation_input.input_detection_segmentation_result.segmentation_output.masks[
                    i
                ][0]
                for i in range(n)
            ] + [
                evaluation_input.edited_detection_segmentation_result.segmentation_output.masks[
                    i
                ][0]
                for i in range(m)
            ]
        return masks

    def mask_into_np(self, mask: torch.Tensor) -> NDArray:
        np_mask: NDArray = mask.cpu().numpy().astype(bool)
        return np_mask
