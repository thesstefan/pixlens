import dataclasses
import json
import pathlib
import pprint

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation import utils as image_utils
from pixlens.visualization import annotation


@dataclasses.dataclass
class BackgroundPreservationArtifacts(
    evaluation_interfaces.EvaluationArtifacts,
):
    union_mask_input: Image.Image | None
    union_mask_edited: Image.Image | None

    def persist(self, save_dir: pathlib.Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.union_mask_input:
            self.union_mask_input.save(
                save_dir / "background_union_mask_input.png",
            )
        if self.union_mask_edited:
            self.union_mask_edited.save(
                save_dir / "background_union_mask_edited.png",
            )


@dataclasses.dataclass(kw_only=True)
class BackgroundPreservationOutput(evaluation_interfaces.EvaluationOutput):
    background_score: float = 0.0

    def persist(self, save_dir: pathlib.Path) -> None:
        save_dir = save_dir / "background_preservation"
        save_dir.mkdir(parents=True, exist_ok=True)

        score_summary = {
            "success": self.success,
            "background_score": self.background_score,
        }
        json_str = json.dumps(score_summary, indent=4)
        score_json_path = save_dir / "scores.json"

        with score_json_path.open("w") as score_json:
            score_json.write(json_str)

        if self.artifacts:
            self.artifacts.persist(save_dir)


class BackgroundPreservation(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> BackgroundPreservationOutput:
        input_image = evaluation_input.input_image
        edited_image = evaluation_input.edited_image
        if input_image.size != edited_image.size:
            edited_image = edited_image.resize(
                input_image.size,
                Image.Resampling.LANCZOS,
            )
        masks = self.get_masks(evaluation_input)
        np_masks = [self.mask_into_np(mask) for mask in masks]
        reshaped_masks = []
        for mask in np_masks:
            if mask.shape != masks[0].shape:
                new_mask = image_utils.resize_mask(mask, np_masks[0])
                reshaped_masks.append(new_mask)
            else:
                reshaped_masks.append(mask)
        union_mask = image_utils.compute_union_segmentation_masks(
            reshaped_masks,
        )
        score = image_utils.get_normalized_background_score(
            1
            - (
                image_utils.compute_mse_over_mask(
                    input_image,
                    edited_image,
                    union_mask,
                    union_mask,
                    background=True,
                    gray_scale=False,
                )
                + image_utils.compute_mse_over_mask(
                    input_image,
                    edited_image,
                    union_mask,
                    union_mask,
                    background=True,
                    gray_scale=True,
                )
            )
            / 2,
        )
        if score == -1:
            return BackgroundPreservationOutput(
                success=False,
                edit_specific_score=0,
            )
        return BackgroundPreservationOutput(
            success=True,
            edit_specific_score=0,
            background_score=score,
            artifacts=BackgroundPreservationArtifacts(
                union_mask_input=annotation.annotate_mask(
                    masks=torch.tensor(union_mask).view(
                        [1, 1, union_mask.shape[0], union_mask.shape[1]],
                    ),
                    image=input_image,
                    mask_alpha=1,
                    color_mask=np.array([0, 0, 0]),
                ),
                union_mask_edited=annotation.annotate_mask(
                    masks=torch.tensor(union_mask).view(
                        [1, 1, union_mask.shape[0], union_mask.shape[1]],
                    ),
                    image=edited_image,
                    mask_alpha=1,
                    color_mask=np.array([0, 0, 0]),
                ),
            ),
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
            edit_type_class.OBJECT_REMOVAL,
        ]
        masks = [torch.zeros(evaluation_input.input_image.size).T]
        if edit_type in add_type:
            indices = image_utils.find_word_indices(
                evaluation_input.edited_detection_segmentation_result.detection_output.phrases,
                evaluation_input.updated_strings.to_attribute,
            )
            masks += [
                evaluation_input.edited_detection_segmentation_result.segmentation_output.masks[
                    index
                ][0]  # .reshape(evaluation_input.edited_image.size)
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
                ][0]  # .reshape(evaluation_input.input_image.size)
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
