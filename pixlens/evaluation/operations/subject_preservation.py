import dataclasses
import logging
import pathlib
import pprint

import cv2  # type: ignore[import]
import numpy as np
import torch
from PIL import Image

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import utils
from pixlens.evaluation.interfaces import (
    EvaluationArtifacts,
    EvaluationInput,
    EvaluationOutput,
    OperationEvaluation,
)
from pixlens.evaluation.operations.visualization import plotting
from pixlens.visualization import annotation
from pixlens.visualization.plotting import figure_to_image


@dataclasses.dataclass
class SubjectPreservationArtifacts(EvaluationArtifacts):
    sift_matches: Image.Image | None
    color_hist_plots: Image.Image | None
    position_visualization: Image.Image | None

    def persist(self, save_dir: pathlib.Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.sift_matches:
            self.sift_matches.save(save_dir / "sift.png")

        if self.color_hist_plots:
            self.color_hist_plots.save(save_dir / "color_histogram.png")

        if self.position_visualization:
            self.position_visualization.save(save_dir / "position_diff.png")


@dataclasses.dataclass(kw_only=True)
class SubjectPreservationOutput(EvaluationOutput):
    sift_score: float = 0.0
    color_score: float = 0.0
    position_score: float = 0.0

    def persist(self, save_dir: pathlib.Path) -> None:
        save_dir = save_dir / "subject_preservation"
        save_dir.mkdir(parents=True, exist_ok=True)

        score_summary = {
            "success": self.success,
            "sift_score": self.sift_score,
            "color_score": self.color_score,
            "position_score": self.position_score,
        }
        json_str = pprint.pformat(score_summary, compact=True).replace("'", '"')
        score_json_path = save_dir / "scores.json"

        with score_json_path.open("w") as score_json:
            score_json.write(json_str)

        if self.artifacts:
            self.artifacts.persist(save_dir)


class SubjectPreservation(OperationEvaluation):
    color_hist_bins: int
    sift_min_matches: int
    sift: cv2.SIFT  # type: ignore[no-any-unimported]
    flann_matcher: cv2.FlannBasedMatcher  # type: ignore[no-any-unimported]

    def __init__(
        self,
        color_hist_bins: int = 32,
        sift_min_matches: int = 5,
        # See section 7.1 in SIFT paper (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
        sift_distance_ratio: float = 0.75,
    ) -> None:
        self.color_hist_bins = color_hist_bins
        self.sift_min_matches = sift_min_matches
        self.sift_distance_ratio = sift_distance_ratio

        self.sift = cv2.SIFT_create()
        self.flann_matcher = cv2.FlannBasedMatcher(
            {
                "algorithm": 1,  # FLANN_INDEX_KDTREE
                "trees": 5,
            },
            {"checks": 50},
        )

    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> SubjectPreservationOutput:
        category = evaluation_input.updated_strings.category

        category_in_input = get_detection_segmentation_result_of_target(
            evaluation_input.input_detection_segmentation_result,
            category,
        )

        category_in_edited = get_detection_segmentation_result_of_target(
            evaluation_input.edited_detection_segmentation_result,
            category,
        )

        if len(category_in_input.detection_output.phrases) == 0:
            logging.warning("No %s detected in the input image", category)
            return SubjectPreservationOutput(
                edit_specific_score=0,
                success=False,
            )

        if len(category_in_edited.detection_output.phrases) == 0:
            logging.warning("No %s detected in edited image", category)
            return SubjectPreservationOutput(
                edit_specific_score=0,
                success=False,
            )

        category_mask_edited = torch.Tensor(
            utils.pad_into_shape_2d(
                category_in_edited.segmentation_output.masks[0].cpu().numpy(),
                category_in_input.segmentation_output.masks[0].shape,
            ),
        )

        sift_score, sift_visualization = self.compute_sift_score(
            evaluation_input.input_image,
            evaluation_input.edited_image,
            category_in_input.segmentation_output.masks[0],
            category_mask_edited,
        )

        color_score, color_histogram_visualization = self.compute_color_score(
            evaluation_input.input_image,
            evaluation_input.edited_image,
            category_in_input.segmentation_output.masks[0],
            category_mask_edited,
        )

        position_score, position_visualization = self.compute_position_score(
            evaluation_input.input_image,
            category_in_input.segmentation_output.masks[0],
            category_mask_edited,
        )

        ssim_score = self.compute_ssim_score(
            evaluation_input.input_image,
            evaluation_input.edited_image,
            category_in_input.segmentation_output.masks[0],
            category_mask_edited,
        )

        return SubjectPreservationOutput(
            success=True,
            edit_specific_score=0,
            sift_score=sift_score,
            color_score=color_score,
            position_score=position_score,
            ssim_score=ssim_score,
            artifacts=SubjectPreservationArtifacts(
                sift_visualization,
                color_histogram_visualization,
                position_visualization,
            ),
        )

    def compute_sift_score(
        self,
        input_image: Image.Image,
        edited_image: Image.Image,
        input_mask: torch.Tensor,
        edited_mask: torch.Tensor,
    ) -> tuple[float, Image.Image | None]:
        input_cv_image = cv2.cvtColor(
            np.array(input_image),
            cv2.COLOR_RGB2GRAY,
        )
        edited_cv_image = cv2.cvtColor(
            np.array(edited_image),
            cv2.COLOR_RGB2GRAY,
        )

        # TODO: Handle multiple objects
        input_keypoints, input_descriptors = self.sift.detectAndCompute(
            input_cv_image,
            np.squeeze(input_mask.cpu().numpy().astype(np.uint8)) * 255,
        )
        edited_keypoints, edited_descriptors = self.sift.detectAndCompute(
            edited_cv_image,
            np.squeeze(edited_mask.cpu().numpy().astype(np.uint8)) * 255,
        )

        matches = self.flann_matcher.knnMatch(
            input_descriptors,
            edited_descriptors,
            k=2,
        )

        good_matches = tuple(
            (A, B)
            for A, B in matches
            if A.distance < self.sift_distance_ratio * B.distance
        )

        match_visualization = annotation.sift_match_visualization(
            input_image,
            edited_image,
            good_matches,
            input_keypoints,
            edited_keypoints,
        )

        if len(good_matches) < self.sift_min_matches:
            return 0.0, None

        score = len(good_matches) / max(
            len(input_descriptors),
            len(edited_descriptors),
        )

        return score, match_visualization

    def compute_color_score(
        self,
        input_image: Image.Image,
        edited_image: Image.Image,
        input_mask: torch.Tensor,
        edited_mask: torch.Tensor,
    ) -> tuple[float, Image.Image]:
        input_color_hist = utils.compute_color_hist_vector(
            input_image,
            mask=np.squeeze(input_mask.cpu().numpy().astype(np.uint8)),
            bins=self.color_hist_bins,
        )
        edited_color_hist = utils.compute_color_hist_vector(
            edited_image,
            mask=np.squeeze(edited_mask.cpu().numpy().astype(np.uint8)),
            bins=self.color_hist_bins,
        )

        normalized_input_color_hist = input_color_hist / (
            input_image.width * input_image.height
        )
        normalized_edited_color_hist = edited_color_hist / (
            edited_image.width * edited_image.height
        )

        color_histogram_figure = plotting.plot_color_histograms(
            np.stack(
                [normalized_input_color_hist, normalized_edited_color_hist],
            ),
        )
        score = utils.cosine_similarity(
            normalized_input_color_hist,
            normalized_edited_color_hist,
        )

        return score, figure_to_image(color_histogram_figure)

    def compute_iou_score(
        self,
        input_mask: torch.Tensor,
        edited_mask: torch.Tensor,
    ) -> float:
        # TODO: Implement aligned IoU
        raise NotImplementedError

    def compute_ssim_score(
        self,
        input_image: Image.Image,
        edited_image: Image.Image,
        input_mask: torch.Tensor,
        edited_mask: torch.Tensor,
    ) -> float:
        return utils.compute_ssim_over_mask(
            input_image,
            edited_image,
            input_mask.cpu().numpy(),
            edited_mask.cpu().numpy(),
        )

    def compute_position_score(
        self,
        input_image: Image.Image,
        input_mask: torch.Tensor,
        edited_mask: torch.Tensor,
    ) -> tuple[float, Image.Image]:
        # TODO: Make utils.center_of_mass return a np.array
        input_center = np.array(utils.center_of_mass(input_mask))
        edited_center = np.array(utils.center_of_mass(edited_mask))

        visualization_image = annotation.annotate_mask(
            input_mask,
            input_image,
        )
        visualization_image = annotation.draw_center_of_masses(
            visualization_image,
            (input_center[0], input_center[1]),
            (edited_center[0], edited_center[1]),
        )

        normalized_input_center = np.divide(
            input_center,
            np.squeeze(input_mask).shape,
        )
        normalized_edit_center = np.divide(
            edited_center,
            np.squeeze(edited_mask).shape,
        )

        score = np.linalg.norm(
            normalized_input_center - normalized_edit_center,
        ).item()

        return score, visualization_image
