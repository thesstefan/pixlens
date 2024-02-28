from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series
from PIL import Image

from pixlens.dataset.edit_dataset import EditDataset
from pixlens.detection import interfaces as detection_interfaces
from pixlens.detection.utils import get_separator
from pixlens.editing.interfaces import (
    ImageEditingPromptType,
    PromptableImageEditingModel,
)
from pixlens.evaluation import interfaces
from pixlens.evaluation.interfaces import EditType
from pixlens.evaluation.operations.background_preservation import (
    BackgroundPreservationOutput,
)
from pixlens.evaluation.operations.subject_preservation import (
    SubjectPreservationOutput,
)
from pixlens.evaluation.utils import get_clean_to_attribute_for_detection
from pixlens.utils.utils import get_cache_dir, get_image_extension
from pixlens.visualization import annotation


class EvaluationSchema(pa.DataFrameModel):
    edit_id: Series[int] = pa.Field(ge=0)
    edit_type: Series[str] = pa.Field(
        isin=[edit_type.value for edit_type in EditType],
    )
    model_id: Series[str]
    evaluation_success: Series[bool]
    edit_specific_score: Series[float]
    subject_preservation_success: Series[float]
    subject_sift_score: Series[float]
    subject_color_score: Series[float]
    subject_position_score: Series[float]
    subject_ssim_score: Series[float]
    subject_aligned_iou: Series[float]
    background_preservation_success: Series[bool]
    background_score: Series[float]


class EvaluationPipeline:
    def __init__(
        self,
        edit_dataset: EditDataset,
    ) -> None:
        self.edit_dataset = edit_dataset
        self.detection_model: detection_interfaces.PromptDetectAndBBoxSegmentModel  # noqa: E501
        self.editing_models: list[PromptableImageEditingModel]
        self.evaluation_dataset: pa.typing.DataFrame[EvaluationSchema]
        self.initialize_evaluation_dataset()

    def initialize_evaluation_dataset(self) -> None:
        self.evaluation_dataset = pd.DataFrame(
            columns=[
                "edit_id",
                "edit_type",
                "model_id",
                "evaluation_success",
                "edit_specific_score",
                "subject_preservation_success",
                "subject_sift_score",
                "subject_color_score",
                "subject_position_score",
                "subject_ssim_score",
                "subject_aligned_iou",
                "background_preservation_success",
                "background_score",
            ],
        )

    def get_input_image_from_edit_id(self, edit_id: int) -> Image.Image:
        image_path = Path(self.edit_dataset.get_edit(edit_id).image_path)
        image_extension = get_image_extension(image_path)
        if image_extension:
            return Image.open(image_path.with_suffix(image_extension))
        raise FileNotFoundError

    def get_edited_image_from_edit(
        self,
        edit: interfaces.Edit,
        prompt: str,
        model: PromptableImageEditingModel,
    ) -> Image.Image:
        edit_path = (
            get_cache_dir()
            / model.model_id
            / Path(edit.image_path).stem
            / prompt
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

    def init_editing_models(
        self,
        model_list: list[PromptableImageEditingModel],
    ) -> None:
        self.editing_models = model_list

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
        editing_model: PromptableImageEditingModel,
    ) -> interfaces.EvaluationInput:
        prompt = (
            edit.instruction_prompt
            if editing_model.prompt_type == ImageEditingPromptType.INSTRUCTION
            else edit.description_prompt
        )
        input_image = self.get_input_image_from_edit_id(edit.edit_id)
        edited_image = self.get_edited_image_from_edit(
            edit,
            prompt,
            editing_model,
        )
        from_attribute = (
            None if pd.isna(edit.from_attribute) else edit.from_attribute
        )
        if not pd.isna(edit.to_attribute):
            edit.to_attribute = "".join(
                char if char.isalpha() or char.isspace() else " "
                for char in edit.to_attribute
            )
            filtered_to_attribute = get_clean_to_attribute_for_detection(edit)
        else:
            edit.to_attribute = None
            filtered_to_attribute = None
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
        if edit.edit_type == "alter_parts":
            # if alter_parts, we need to do detection and segmentation
            # on the edited image ONLY with the to_attribute
            list_for_det_seg = [
                filtered_to_attribute
                if filtered_to_attribute is not None
                else " ",
            ]
            list_for_det_seg = list(set(list_for_det_seg))
            separator = get_separator(self.detection_model)
            prompt_for_det_seg = separator.join(list_for_det_seg)

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

    def update_evaluation_dataset(
        self,
        edit: interfaces.Edit,
        model_id: str,
        evaluation_outputs: list[interfaces.EvaluationOutput],
    ) -> None:
        for evaluation_output in evaluation_outputs:
            if isinstance(
                evaluation_output,
                SubjectPreservationOutput,
            ):
                last_row_index = self.evaluation_dataset.index[-1]
                self.evaluation_dataset.loc[
                    last_row_index,
                    "subject_preservation_success",
                ] = evaluation_output.success
                self.evaluation_dataset.loc[
                    last_row_index,
                    "subject_sift_score",
                ] = evaluation_output.sift_score
                self.evaluation_dataset.loc[
                    last_row_index,
                    "subject_color_score",
                ] = evaluation_output.color_score
                self.evaluation_dataset.loc[
                    last_row_index,
                    "subject_position_score",
                ] = evaluation_output.position_score
                self.evaluation_dataset.loc[
                    last_row_index,
                    "subject_ssim_score",
                ] = evaluation_output.ssim_score
                self.evaluation_dataset.loc[
                    last_row_index,
                    "subject_aligned_iou",
                ] = evaluation_output.aligned_iou
            elif isinstance(
                evaluation_output,
                BackgroundPreservationOutput,
            ):
                last_row_index = self.evaluation_dataset.index[-1]
                self.evaluation_dataset.loc[
                    last_row_index,
                    "background_preservation_success",
                ] = evaluation_output.success
                self.evaluation_dataset.loc[
                    last_row_index,
                    "background_score",
                ] = evaluation_output.background_score
            elif isinstance(evaluation_output, interfaces.EvaluationOutput):
                # order of IF statements matters, because EvaluationOutput
                # is a superclass of SubjectPreservationOutput and
                # BackgroundPreservationOutput
                self.evaluation_dataset.loc[len(self.evaluation_dataset)] = {
                    "edit_id": edit.edit_id,
                    "edit_type": edit.edit_type,
                    "model_id": model_id,
                    "evaluation_success": evaluation_output.success,
                    "edit_specific_score": evaluation_output.edit_specific_score,  # noqa: E501
                }
            else:
                raise NotImplementedError

    def save_evaluation_dataset(self) -> None:
        self.evaluation_dataset.to_csv(
            Path(get_cache_dir(), "evaluation_results.csv"),
            index=False,
        )

    def load_evaluation_dataset(self) -> None:
        self.evaluation_dataset = EvaluationSchema.validate(
            pd.read_csv(get_cache_dir() / "evaluation_results.csv"),
        )

    def get_aggregated_scores_for_model(
        self,
        model_id: str,
    ) -> dict[str, dict[str, float]]:
        # extract all possible edit types
        edit_types = self.evaluation_dataset["edit_type"].unique()

        # initialize a dictionary to store the aggregated scores
        aggregated_scores = {}
        for edit_type in edit_types:
            aggregated_scores[
                edit_type
            ] = self.get_aggregated_scores_for_model_and_edit_type(
                model_id,
                edit_type,
            )
        aggregated_scores["overall_mean"] = self.get_mean_scores_for_model(
            model_id,
        )

        return aggregated_scores

    def get_aggregated_scores_for_edit_type(
        self,
    ) -> dict[str, dict[str, float]]:
        edit_types = self.evaluation_dataset["edit_type"].unique()
        model_ids = self.evaluation_dataset["model_id"].unique()

        results = {}
        for edit_type in edit_types:
            aggregated_scores = {}
            for model_id in model_ids:
                aggregated_scores[
                    model_id
                ] = self.get_aggregated_scores_for_model_and_edit_type(
                    model_id,
                    edit_type,
                )
            aggregated_scores[
                "overall_mean"
            ] = self.get_mean_scores_for_edit_type(
                edit_type,
            )
            results[edit_type] = aggregated_scores
        return results

    def get_aggregated_scores_for_model_and_edit_type(
        self,
        model_id: str,
        edit_type: str,
    ) -> dict[str, float]:
        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["model_id"] == model_id)
            & (self.evaluation_dataset["edit_type"] == edit_type)
            & (self.evaluation_dataset["evaluation_success"])
        ]
        edit_specific_score = filtered_evaluation_dataset[
            "edit_specific_score"
        ].mean()

        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["model_id"] == model_id)
            & (self.evaluation_dataset["edit_type"] == edit_type)
            & (self.evaluation_dataset["subject_preservation_success"])
        ]
        sift_score = filtered_evaluation_dataset["subject_sift_score"].mean()
        color_score = filtered_evaluation_dataset["subject_color_score"].mean()
        position_score = filtered_evaluation_dataset[
            "subject_position_score"
        ].mean()
        aligned_iou = filtered_evaluation_dataset["subject_aligned_iou"].mean()

        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["model_id"] == model_id)
            & (self.evaluation_dataset["edit_type"] == edit_type)
            & (self.evaluation_dataset["background_preservation_success"])
        ]
        background_score = filtered_evaluation_dataset[
            "background_score"
        ].mean()

        results = {
            "edit_specific_score": edit_specific_score,
            "subject_sift_score": sift_score,
            "subject_color_score": color_score,
            "subject_position_score": position_score,
            "subject_aligned_iou": aligned_iou,
            "background_score": background_score,
        }

        for key, value in results.items():
            if pd.isna(value):
                results[key] = None
        return results

    def get_mean_scores_for_model(
        self,
        model_id: str,
    ) -> dict[str, float]:
        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["model_id"] == model_id)
            & (self.evaluation_dataset["evaluation_success"])
        ]
        edit_specific_score = filtered_evaluation_dataset[
            "edit_specific_score"
        ].mean()

        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["model_id"] == model_id)
            & (self.evaluation_dataset["subject_preservation_success"])
        ]
        sift_score = filtered_evaluation_dataset["subject_sift_score"].mean()

        # before computing the color mean score,
        # filter out the rows with edit_type == "color"
        color_filtered_evaluation_dataset = filtered_evaluation_dataset[
            filtered_evaluation_dataset["edit_type"] != EditType.COLOR
        ]
        color_score = color_filtered_evaluation_dataset[
            "subject_color_score"
        ].mean()

        # before computing the position mean score,
        # filter out the rows with edit_type == "position_replacement"
        position_filtered_evaluation_dataset = filtered_evaluation_dataset[
            filtered_evaluation_dataset["edit_type"]
            != EditType.POSITION_REPLACEMENT
        ]
        position_score = position_filtered_evaluation_dataset[
            "subject_position_score"
        ].mean()

        # before computing the aligned_iou mean score,
        # filter out the rows with edit_type == "size"
        aligned_iou_filtered_evaluation_dataset = filtered_evaluation_dataset[
            filtered_evaluation_dataset["edit_type"] != EditType.SIZE
        ]
        aligned_iou = aligned_iou_filtered_evaluation_dataset[
            "subject_aligned_iou"
        ].mean()

        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["model_id"] == model_id)
            & (self.evaluation_dataset["background_preservation_success"])
        ]
        background_score = filtered_evaluation_dataset[
            "background_score"
        ].mean()

        results = {
            "edit_specific_score": edit_specific_score,
            "subject_sift_score": sift_score,
            "subject_color_score": color_score,
            "subject_position_score": position_score,
            "subject_aligned_iou": aligned_iou,
            "background_score": background_score,
        }

        for key, value in results.items():
            if pd.isna(value):
                results[key] = None
        return results

    def get_mean_scores_for_edit_type(
        self,
        edit_type: str,
    ) -> dict[str, float]:
        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["edit_type"] == edit_type)
            & (self.evaluation_dataset["evaluation_success"])
        ]
        edit_specific_score = filtered_evaluation_dataset[
            "edit_specific_score"
        ].mean()

        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["edit_type"] == edit_type)
            & (self.evaluation_dataset["subject_preservation_success"])
        ]
        sift_score = filtered_evaluation_dataset["subject_sift_score"].mean()
        color_score = filtered_evaluation_dataset["subject_color_score"].mean()
        position_score = filtered_evaluation_dataset[
            "subject_position_score"
        ].mean()
        aligned_iou = filtered_evaluation_dataset["subject_aligned_iou"].mean()

        filtered_evaluation_dataset = self.evaluation_dataset[
            (self.evaluation_dataset["edit_type"] == edit_type)
            & (self.evaluation_dataset["background_preservation_success"])
        ]
        background_score = filtered_evaluation_dataset[
            "background_score"
        ].mean()

        results = {
            "edit_specific_score": edit_specific_score,
            "subject_sift_score": sift_score,
            "subject_color_score": color_score,
            "subject_position_score": position_score,
            "subject_aligned_iou": aligned_iou,
            "background_score": background_score,
        }

        for key, value in results.items():
            if pd.isna(value):
                results[key] = None
        return results
