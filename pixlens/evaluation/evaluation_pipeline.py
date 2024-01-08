import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pixlens.detection import interfaces as detection_interfaces
from pixlens.detection.utils import get_separator
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation import interfaces
from pixlens.evaluation.utils import get_clean_to_attribute_for_detection
from pixlens.utils.utils import get_cache_dir, get_image_extension
from pixlens.visualization import annotation


class EvaluationPipeline:
    def __init__(self, device: torch.device) -> None:
        self.device = "cpu"  # original was cuda
        self.edit_dataset: pd.DataFrame
        self.get_edit_dataset()
        self.detection_model: detection_interfaces.PromptDetectAndBBoxSegmentModel  # noqa: E501
        self.editing_model: PromptableImageEditingModel

    def get_edit_dataset(self) -> None:
        pandas_path = Path(get_cache_dir(), "edit_dataset.csv")
        if pandas_path.exists():
            self.edit_dataset = pd.read_csv(pandas_path)
        else:
            logging.error(f"Edit dataset ({pandas_path}) not cached")  # noqa: G004
            raise FileNotFoundError

    def get_input_image_from_edit_id(self, edit_id: int) -> Image.Image:
        image_path = self.edit_dataset.iloc[edit_id]["input_image_path"]
        image_path = Path(image_path)
        image_extension = get_image_extension(image_path)
        if image_extension:
            return Image.open(image_path.with_suffix(image_extension))
        raise FileNotFoundError

    def get_edited_image_from_edit(
        self,
        edit: interfaces.Edit,
        model: PromptableImageEditingModel,
    ) -> Image.Image:
        prompt = model.generate_prompt(edit)
        edit_path = Path(
            get_cache_dir(),
            "models--" + model.get_model_name(),
            f"000000{edit.image_id!s:>06}",
            prompt,
        )
        extension = get_image_extension(edit_path)
        if extension:
            return Image.open(edit_path.with_suffix(extension))
        raise FileNotFoundError

    def init_detection_model(
        self,
        model: detection_interfaces.PromptDetectAndBBoxSegmentModel,
    ) -> None:
        self.detection_model = model

    def init_editing_model(self, model: PromptableImageEditingModel) -> None:
        self.editing_model = model

    def do_detection_and_segmentation(
        self,
        image: Image.Image,
        prompt: str,
    ) -> detection_interfaces.DetectionSegmentationResult:
        (
            segmentation_output,
            detection_output,
        ) = self.detection_model.detect_and_segment(prompt=prompt, image=image)
        return detection_interfaces.DetectionSegmentationResult(
            detection_output=detection_output,
            segmentation_output=segmentation_output,
        )

    def get_all_inputs_for_edit(
        self,
        edit: interfaces.Edit,
    ) -> interfaces.EvaluationInput:
        input_image = self.get_input_image_from_edit_id(edit.edit_id)
        edited_image = self.get_edited_image_from_edit(edit, self.editing_model)
        prompt = self.editing_model.generate_prompt(edit)
        from_attribute = (
            None if pd.isna(edit.from_attribute) else edit.from_attribute
        )
        edit.to_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in edit.to_attribute
        )
        filtered_to_attribute = get_clean_to_attribute_for_detection(edit)
        category = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in edit.category
        )
        list_for_det_seg = [
            item
            for item in [category, from_attribute, filtered_to_attribute]
            if item is not None
        ]

        list_for_det_seg = list(set(list_for_det_seg))

        separator = get_separator(self.detection_model)
        prompt_for_det_seg = separator.join(list_for_det_seg)

        input_detection_segmentation_result = (
            self.do_detection_and_segmentation(
                input_image,
                prompt_for_det_seg,
            )
        )
        edited_detection_segmentation_result = (
            self.do_detection_and_segmentation(
                edited_image,
                prompt_for_det_seg,
            )
        )

        # Input image
        if input_detection_segmentation_result.detection_output.bounding_boxes.any():
            annotated_input_image = annotation.annotate_detection_output(
                np.asarray(input_image),
                input_detection_segmentation_result.detection_output,
            )

            if input_detection_segmentation_result.segmentation_output.masks.any():
                masked_annotated_input_image = annotation.annotate_mask(
                    input_detection_segmentation_result.segmentation_output.masks,
                    annotated_input_image,
                )
            else:
                masked_annotated_input_image = annotated_input_image
        else:
            annotated_input_image = input_image
            masked_annotated_input_image = input_image

        # Edited image
        if edited_detection_segmentation_result.detection_output.bounding_boxes.any():
            annotated_edited_image = annotation.annotate_detection_output(
                np.asarray(edited_image),
                edited_detection_segmentation_result.detection_output,
            )

            if edited_detection_segmentation_result.segmentation_output.masks.any():
                masked_annotated_edited_image = annotation.annotate_mask(
                    edited_detection_segmentation_result.segmentation_output.masks,
                    annotated_edited_image,
                )
            else:
                masked_annotated_edited_image = annotated_edited_image
        else:
            annotated_edited_image = edited_image
            masked_annotated_edited_image = edited_image

        return interfaces.EvaluationInput(
            input_image=input_image,
            edited_image=edited_image,
            annotated_input_image=masked_annotated_input_image,
            annotated_edited_image=masked_annotated_edited_image,
            prompt=prompt,
            input_detection_segmentation_result=input_detection_segmentation_result,
            edited_detection_segmentation_result=edited_detection_segmentation_result,
            edit=edit,
            updated_strings=interfaces.UpdatedStrings(
                category=category,
                from_attribute=from_attribute,
                to_attribute=filtered_to_attribute,
            ),
        )

    def get_all_scores_for_edit(
        self,
        edit: interfaces.Edit,
    ) -> dict[str, float]:
        evaluation_input = self.get_all_inputs_for_edit(edit)
        edit_type_dependent_scores = self.get_edit_dependent_scores_for_edit(
            evaluation_input,
        )
        edit_type_indpendent_scores = self.get_edit_independent_scores_for_edit(
            evaluation_input,
        )
        return {**edit_type_dependent_scores, **edit_type_indpendent_scores}

    def get_edit_dependent_scores_for_edit(
        self,
        evaluation_input: interfaces.EvaluationInput,
    ) -> dict[str, float]:
        raise NotImplementedError

    def get_edit_independent_scores_for_edit(
        self,
        evaluation_input: interfaces.EvaluationInput,
    ) -> dict[str, float]:
        raise NotImplementedError
