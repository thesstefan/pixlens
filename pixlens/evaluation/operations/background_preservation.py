import torch
from numpy.typing import NDArray

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation import utils as image_utils


class BackgroundPreservation(evaluation_interfaces.GeneralEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> float:
        input_image = evaluation_input.input_image
        edited_image = evaluation_input.edited_image
        masks = self.get_masks(evaluation_input)
        union_mask = image_utils.compute_union_segmentation_masks(masks)
        return image_utils.compute_ssim_over_mask(
            input_image,
            edited_image,
            self.mask_into_np(union_mask),
            self.mask_into_np(union_mask),
            background=True,
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
            edit_type_class.OBJECT_REPLACEMENT,
        ]
        only_category_type = [
            edit_type_class.OBJECT_REPLACEMENT,
            edit_type_class.TEXTURE,
            edit_type_class.COLOR,
            edit_type_class.SIZE,
            edit_type_class.SHAPE,
            edit_type_class.STYLE,
            edit_type_class.POSITION_REPLACEMENT,
            edit_type_class.VIEWPOINT,
        ]
        if edit_type in add_type:
            indices = image_utils.find_word_indices(
                evaluation_input.edited_detection_segmentation_result.detection_output.phrases,
                evaluation_input.updated_strings.to_attribute,
            )
            masks = [
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
            masks = [
                evaluation_input.input_detection_segmentation_result.segmentation_output.masks[
                    index
                ][0]
                for index in indices
            ]
        else:
            n = evaluation_input.input_detection_segmentation_result.segmentation_output.masks.size()[
                0
            ]
            m = evaluation_input.edited_detection_segmentation_result.segmentation_output.masks.size()[
                0
            ]
            masks = [
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
