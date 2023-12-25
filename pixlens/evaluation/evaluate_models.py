from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from pixlens.detection import interfaces as detection_interfaces
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation import interfaces
from pixlens.evaluation.edit_dataset import PreprocessingPipeline
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
            self.do_detection_and_segmentation(input_image, edit.category)
        )
        edited_detection_segmentation_result = (
            self.do_detection_and_segmentation(edited_image, edit.to_attribute)
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
