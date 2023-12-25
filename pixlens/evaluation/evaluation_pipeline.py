from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from pixlens.detection import interfaces as detection_interfaces
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation import interfaces
from pixlens.evaluation.preprocessing_pipeline import PreprocessingPipeline
from pixlens.evaluation.utils import (
    compute_area_ratio,
    compute_iou,
    get_prompt_for_input_detection,
    get_prompt_for_output_detection,
)
from pixlens.utils.utils import get_cache_dir, get_image_extension


class EvaluationPipeline:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.edit_dataset: pd.DataFrame
        self.get_edit_dataset()
        self.detection_model: detection_interfaces.PromptDetectAndBBoxSegmentModel  # noqa: E501
        self.editing_model: PromptableImageEditingModel

    def get_edit_dataset(self) -> None:
        pandas_path = Path(get_cache_dir(), "edit_dataset.csv")
        if pandas_path.exists():
            self.edit_dataset = pd.read_csv(pandas_path)
        else:
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
        prompt = PreprocessingPipeline.generate_prompt(edit)
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
        prompt = PreprocessingPipeline.generate_prompt(edit)
        input_detection_segmentation_result = (
            self.do_detection_and_segmentation(
                input_image,
                get_prompt_for_input_detection(edit),
            )
        )
        edited_detection_segmentation_result = (
            self.do_detection_and_segmentation(
                edited_image,
                get_prompt_for_output_detection(edit),
            )
        )
        return interfaces.EvaluationInput(
            input_image=input_image,
            edited_image=edited_image,
            prompt=prompt,
            input_detection_segmentation_result=input_detection_segmentation_result,
            edited_detection_segmentation_result=edited_detection_segmentation_result,
            edit=edit,
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

    def get_score_for_size_edit(
        self,
        evaluation_input: interfaces.EvaluationInput,
    ) -> float:
        # 1 - Check if object is present in both input and output:
        score_area_ratio = 0.0
        score_iou = 0.0
        if len(
            evaluation_input.edited_detection_segmentation_result.detection_output.phrases,
        ):
            # 2 - Check if resize is small or big and compute area difference
            transformation = evaluation_input.edit.to_attribute
            mask_input = evaluation_input.input_detection_segmentation_result.segmentation_output.masks[
                0
            ]
            idmax = evaluation_input.edited_detection_segmentation_result.segmentation_output.logits.argmax()
            mask_edited = evaluation_input.edited_detection_segmentation_result.segmentation_output.masks[
                idmax
            ]

            area_ratio = compute_area_ratio(mask_input, mask_edited)
            if (transformation == "small" and area_ratio > 1) or (
                transformation == "big" and area_ratio < 1
            ):
                score_area_ratio = 1

            score_iou = compute_iou(mask_input, mask_edited)

            return (score_area_ratio + score_iou) / 2

        return 0  # Object wasn't present at output
