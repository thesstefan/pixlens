import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageColor

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import utils
from pixlens.evaluation.interfaces import (
    EvaluationArtifacts,
    EvaluationInput,
    EvaluationOutput,
    OperationEvaluation,
)
from pixlens.evaluation.multiplicity_resolver import (
    MultiplicityResolution,
    select_one_2d,
)

COLOR_NAME_ALIAS: dict[str, str] = {
    "golden": "gold",
    # "brown": "saddlebrown",
}

COLOR_RBG_ALIAS: dict[str, tuple[int, int, int]] = {
    # ImageColors.get("green") return (0, 128, 0) which skews results
    # quite badly
    "green": (0, 255, 0),
    "brown": (64, 30, 9),  # mix between brown and saddlebrown
}

RNG_SEED = 0


@dataclasses.dataclass
class ColorEditArtifacts(EvaluationArtifacts):
    color_hist_plots: Image.Image | None

    def persist(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.color_hist_plots:
            self.color_hist_plots.save(save_dir / "color_histogram.png")


# TODO: This could apply to all basic operations. Rework the interface
#       to allow for generalization
@dataclasses.dataclass
class ColorEditOutput(EvaluationOutput):
    def persist(self, save_dir: Path) -> None:
        save_dir = save_dir / "color"
        save_dir.mkdir(parents=True, exist_ok=True)

        score_summary = {
            "success": self.success,
            "score": self.edit_specific_score,
        }

        json_str = json.dumps(score_summary, indent=4)
        score_json_path = save_dir / "scores.json"

        with score_json_path.open("w") as score_json:
            score_json.write(json_str)

        if self.artifacts:
            self.artifacts.persist(save_dir)


class ColorEdit(OperationEvaluation):
    color_hist_bins: int
    hist_cmp_method: utils.HistogramComparisonMethod
    category_input_resolution: MultiplicityResolution
    category_edited_resolution: MultiplicityResolution
    rng: np.random.Generator

    def __init__(  # noqa: PLR0913
        self,
        color_hist_bins: int = 32,
        hist_cmp_method: utils.HistogramComparisonMethod = (
            utils.HistogramComparisonMethod.CORRELATION
        ),
        category_input_resolution: MultiplicityResolution = (
            MultiplicityResolution.LARGEST
        ),
        category_edited_resolution: MultiplicityResolution = (
            MultiplicityResolution.CLOSEST
        ),
        synthetic_sigma: float = 75.0,
    ) -> None:
        self.color_hist_bins = color_hist_bins
        self.hist_cmp_method = hist_cmp_method
        self.category_input_resolution = category_input_resolution
        self.category_edited_resolution = category_edited_resolution
        self.synthetic_sigma = synthetic_sigma

        self.rng = np.random.default_rng(RNG_SEED)

    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationOutput:
        category = evaluation_input.updated_strings.category
        target_color = evaluation_input.edit.to_attribute

        # Some prompts contain "golden", while "gold" is the equivalent color
        # provided by PIL.ImageColors
        target_color = COLOR_NAME_ALIAS.get(target_color, target_color)
        category_in_input = get_detection_segmentation_result_of_target(
            evaluation_input.input_detection_segmentation_result,
            category,
        )

        if len(category_in_input.detection_output.phrases) == 0:
            logging.warning("No %s detected in the input image", category)
            return ColorEditOutput(
                edit_specific_score=0,
                success=False,
            )

        selected_category_idx_in_input = select_one_2d(
            category_in_input.segmentation_output.masks.cpu().numpy(),
            self.category_input_resolution,
            confidences=category_in_input.detection_output.logits.cpu().numpy(),
            relative_mask=None,
        )
        category_mask_input = np.squeeze(
            (
                category_in_input.segmentation_output.masks[
                    selected_category_idx_in_input
                ]
            )
            .cpu()
            .numpy(),
        )

        category_in_edited = get_detection_segmentation_result_of_target(
            evaluation_input.edited_detection_segmentation_result,
            category,
        )

        if len(category_in_edited.detection_output.phrases) == 0:
            logging.warning("No %s detected in edited image", category)
            return ColorEditOutput(
                edit_specific_score=0,
                success=False,
            )

        selected_category_idx_in_edited_idx = select_one_2d(
            category_in_edited.segmentation_output.masks.cpu().numpy(),
            self.category_edited_resolution,
            confidences=category_in_edited.detection_output.logits.cpu().numpy(),
            relative_mask=category_mask_input,
        )
        category_mask_edited = np.squeeze(
            (
                category_in_edited.segmentation_output.masks[
                    selected_category_idx_in_edited_idx
                ]
            )
            .cpu()
            .numpy(),
        )

        color_score, color_histogram_image = self.compute_color_score(
            evaluation_input.edited_image,
            category_mask_edited,
            target_color,
        )

        return ColorEditOutput(
            success=True,
            edit_specific_score=color_score,
            artifacts=ColorEditArtifacts(color_histogram_image),
        )

    def compute_color_score(
        self,
        image: Image.Image,
        mask: npt.NDArray[np.bool_],
        target_color: str,
    ) -> tuple[float, Image.Image]:
        target_rgb = COLOR_RBG_ALIAS.get(
            target_color,
            ImageColor.getrgb(target_color),
        )

        synthetic_points = self.rng.multivariate_normal(
            mean=target_rgb,
            cov=np.eye(3, 3) * self.synthetic_sigma,
            size=(image.height, image.width),
        )

        synthetic_points = np.clip(synthetic_points, 0, 255)
        synthetic_image = Image.fromarray(
            synthetic_points.astype(np.uint8),
            "RGB",
        )

        return utils.compare_color_histograms(
            image,
            mask,
            synthetic_image,
            mask,
            method=self.hist_cmp_method,
            num_bins=self.color_hist_bins,
        )
