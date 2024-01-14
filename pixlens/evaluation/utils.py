import enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve

# import delta e color similarity function
from skimage.metrics import structural_similarity as ssim

from pixlens.evaluation.interfaces import Edit, EditType, EvaluationInput
from pixlens.evaluation.operations.visualization import plotting
from pixlens.visualization.plotting import figure_to_image

directions_and_instructions = [
    "add",
    "to",
    "on",
    "top",
    "right",
    "left",
    "below",
    # "toppings",
]
edits = list(EditType)
new_object = ["object_addition", "object_replacement", "background", "texture"]
new_object_with_indication = ["alter_parts", "positional_addition"]
same_object = [
    edit
    for edit in edits
    if edit not in new_object + new_object_with_indication
]


def remove_words_from_string(
    input_string: str,
    words_to_remove: list[str],
) -> str:
    if words_to_remove == []:
        words_to_remove = directions_and_instructions
    # Split the string into words
    words = input_string.split()

    # Remove the words that are in the words_to_remove list
    filtered_words = [word for word in words if word not in words_to_remove]

    # Join the filtered words back into a string
    return " ".join(filtered_words)


def get_clean_to_attribute_for_detection(edit: Edit) -> str | None:
    if edit.to_attribute is np.nan:
        return None
    to_attribute = "".join(
        char if char.isalpha() or char.isspace() else " "
        for char in edit.to_attribute
    )
    if edit.edit_type in new_object:
        return to_attribute
    if edit.edit_type in new_object_with_indication:
        return remove_words_from_string(
            to_attribute,
            directions_and_instructions,
        )
    return None


def compute_area(tensor1: torch.Tensor) -> float:
    area1 = torch.sum(tensor1)
    return area1.item()


def compute_area_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> float:
    area1 = compute_area(numerator)
    area2 = compute_area(denominator)
    if not area2 > 0:
        raise ZeroDivisionError
    return area1 / area2


def is_small_area_within_big_area(
    input_mask: torch.Tensor,
    edited_mask: torch.Tensor,
    confidence_threshold: float = 0.9,
) -> bool:
    # infer small and big area from input and edited masks
    # SD doesn't preserve sizes, rounds up to multiples of 2^n for a given n.
    if input_mask.size(1) > edited_mask.size(1):
        height_pad = input_mask.size(1) - edited_mask.size(1)
        # Pad the bottom of the mask
        edited_mask = F.pad(edited_mask, (0, 0, 0, height_pad), value=False)

    if input_mask.size(2) > edited_mask.size(2):
        width_pad = input_mask.size(2) - edited_mask.size(2)
        # Pad the right side of the mask
        edited_mask = F.pad(edited_mask, (0, width_pad, 0, 0), value=False)
    if compute_area(input_mask) > compute_area(edited_mask):
        small_area = edited_mask
        big_area = input_mask
    else:
        small_area = input_mask
        big_area = edited_mask
    intersection = torch.logical_and(small_area, big_area).sum()
    return intersection.item() / compute_area(small_area) > confidence_threshold


def apply_mask(
    np_image: NDArray,
    mask: NDArray,
    *,
    color_integer: int = 0,
) -> NDArray:
    # Ensure the mask is a boolean array
    mask = mask.astype(bool)
    if mask.shape != np_image.shape[:2]:
        mask = mask.T

    # Apply the mask to each channel
    masked_image = np.zeros_like(np_image)
    for i in range(
        np_image.shape[2],
    ):  # Assuming image has shape [Height, Width, Channels]
        masked_image[:, :, i] = np.where(
            ~mask,
            color_integer,
            np_image[:, :, i],
        )

    return masked_image


def compute_ssim_over_mask(
    input_image: Image.Image,
    edited_image: Image.Image,
    mask1: NDArray,
    mask2: NDArray | None = None,
    *,
    background: bool = False,
) -> float:
    input_image_array = np.array(input_image)
    edited_image_array = np.array(edited_image)

    if edited_image_array.shape != input_image_array.shape:
        edited_image_resized = edited_image.resize(
            input_image.size,
            Image.Resampling.LANCZOS,
        )
        edited_image_array = np.array(edited_image_resized)
        if mask2 is None:
            mask2 = mask1
    input_image_masked = apply_mask(input_image_array, mask1)
    edited_image_masked = apply_mask(edited_image_array, mask2)
    if background:
        input_image_masked = apply_mask(input_image_array, ~mask1)
        edited_image_masked = apply_mask(edited_image_array, ~mask2)
        edited_malicious_masked = apply_mask(
            edited_image_array,
            ~mask2,
            opposite_color=True,
        )
        return (
            float(
                ssim(input_image_masked, edited_image_masked, channel_axis=2),
            )
            + float(
                ssim(
                    input_image_masked,
                    edited_malicious_masked,
                    channel_axis=2,
                ),
            )
        ) / 2
    return float(ssim(input_image_masked, edited_image_masked, channel_axis=2))


def compute_ssim(
    evaluation_input: EvaluationInput,
) -> float:
    input_image_np = np.array(evaluation_input.input_image)
    edited_image_np = np.array(evaluation_input.edited_image)
    if edited_image_np.shape != input_image_np.shape:
        edited_image_resized = evaluation_input.edited_image.resize(
            evaluation_input.input_image.size,
            Image.Resampling.LANCZOS,
        )
        edited_image_np = np.array(edited_image_resized)

    return float(ssim(input_image_np, edited_image_np, channel_axis=2))


def resize_mask(
    mask_to_be_resized: NDArray,
    target_mask: NDArray,
) -> NDArray:
    new_mask = np.zeros_like(target_mask, dtype=bool)
    target_shape = target_mask.shape
    new_mask[
        : min(mask_to_be_resized.shape[0], target_shape[0]),
        : min(mask_to_be_resized.shape[1], target_shape[1]),
    ] = mask_to_be_resized[
        : min(mask_to_be_resized.shape[0], target_shape[0]),
        : min(mask_to_be_resized.shape[1], target_shape[1]),
    ]
    return new_mask


def compute_union_segmentation_masks(masks: list[NDArray]) -> NDArray:
    if not masks:
        raise ValueError("The list of masks cannot be empty")

    union_mask = masks[0]
    for mask in masks[1:]:
        union_mask = np.bitwise_or(union_mask, mask)
    return union_mask


def find_word_indices(
    word_list: list[str],
    target_word: str,
) -> list[int | None]:
    return [
        index for index, word in enumerate(word_list) if word == target_word
    ]


def center_of_mass(segmentation_mask: torch.Tensor) -> tuple[float, float]:
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.arange(segmentation_mask.size(1)),
        torch.arange(segmentation_mask.size(2)),
    )

    # Convert the coordinates to float and move them to the device
    # of the segmentation mask
    y = y.float().to(segmentation_mask.device)
    x = x.float().to(segmentation_mask.device)

    # Multiply coordinates with the segmentation mask
    # to get weighted coordinates
    weighted_y = y * segmentation_mask
    weighted_x = x * segmentation_mask

    # Sum the weighted coordinates along each axis
    sum_y = torch.sum(weighted_y)
    sum_x = torch.sum(weighted_x)

    # Count the total number of True values in the segmentation mask
    total_true_values = torch.sum(segmentation_mask.float())

    # Compute the center of mass
    center_of_mass_y = sum_y / total_true_values
    center_of_mass_x = sum_x / total_true_values

    return center_of_mass_y.item(), center_of_mass_x.item()


def flatten_and_select(
    np_image: NDArray,
    mask: NDArray,
    *,
    channels: int = 3,
) -> NDArray:
    mask = mask.astype(bool)
    # Flatten both image and mask
    if channels == 1:
        flat_image = np_image.reshape(-1)
    else:
        flat_image = np_image.reshape(-1, np_image.shape[-1])
    flat_mask = mask.flatten()
    return flat_image[flat_mask]


def compute_mse_over_mask(
    input_image: Image.Image,
    edited_image: Image.Image,
    mask1: NDArray,
    mask2: NDArray | None = None,
    *,
    background: bool = False,
    gray_scale: bool = False,
) -> float:
    channels = 3
    if gray_scale:
        input_image = input_image.convert("L")
        edited_image = edited_image.convert("L")
        channels = 1
    input_image_array = np.array(input_image)
    edited_image_array = np.array(edited_image)

    # if edited_image_array.shape != input_image_array.shape:
    #     edited_image_resized = edited_image.resize(
    #         input_image.size,
    #         Image.Resampling.LANCZOS,
    #     )
    #     edited_image_array = np.array(edited_image_resized)
    if mask2 is None:
        mask2 = mask1
    if background:
        input_image_masked = (
            flatten_and_select(input_image_array, ~mask1, channels=channels)
            / 255
        )
        edited_image_masked = (
            flatten_and_select(edited_image_array, ~mask2, channels=channels)
            / 255
        )
    else:
        input_image_masked = (
            flatten_and_select(input_image_array, mask1, channels=channels)
            / 255
        )  # normalize
        edited_image_masked = (
            flatten_and_select(edited_image_array, mask2, channels=channels)
            / 255
        )

    return float(np.mean((input_image_masked - edited_image_masked) ** 2))


def compute_mask_intersection(
    whole: torch.Tensor,
    part: torch.Tensor,
) -> float:
    if whole.shape != part.shape:
        # Resize part tensor to match the shape of the whole tensor
        part_float = (
            F.interpolate(
                part.float().unsqueeze(0).unsqueeze(0),
                size=whole.shape,
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
        )
        true_threshold = 0.5
        part = part_float > true_threshold

    intersection = torch.logical_and(
        whole,
        part,
    ).sum()
    part_sum = part.sum()
    if not part_sum > 0:
        raise ZeroDivisionError
    return intersection.item() / part_sum.item()


def compute_bbox_part_whole_ratio(
    whole_bbox: torch.Tensor,
    part_bbox: torch.Tensor,
) -> float:
    # compute the intersection of the bounding boxes
    # where each bbox is a tensor of the form
    # [ymin, xmin, ymax, xmax]

    # compute the intersection of as a ratio of the area of the intersection
    # over the area of the whole_bbox
    intersection = (
        (
            torch.min(whole_bbox[2:], part_bbox[2:])
            - torch.max(whole_bbox[:2], part_bbox[:2])
        )
        .clamp(min=0)
        .prod()
    )
    part_bbox_area = (part_bbox[2:] - part_bbox[:2]).prod()
    if not part_bbox_area > 0:
        raise ZeroDivisionError
    return intersection.item() / part_bbox_area.item()


def flatten_and_select(
    np_image: NDArray,
    mask: NDArray,
    *,
    channels: int = 3,
) -> NDArray:
    mask = mask.astype(bool)
    # Flatten both image and mask
    if channels == 1:
        flat_image = np_image.reshape(-1)
    else:
        flat_image = np_image.reshape(-1, np_image.shape[-1])
    flat_mask = mask.flatten()
    return flat_image[flat_mask]


def compute_mse_over_mask(
    input_image: Image.Image,
    edited_image: Image.Image,
    mask1: NDArray,
    mask2: NDArray | None = None,
    *,
    background: bool = False,
    gray_scale: bool = False,
) -> float:
    channels = 3
    if gray_scale:
        input_image = input_image.convert("L")
        edited_image = edited_image.convert("L")
        channels = 1
    input_image_array = np.array(input_image)
    edited_image_array = np.array(edited_image)

    if edited_image_array.shape != input_image_array.shape:
        edited_image_resized = edited_image.resize(
            input_image.size,
            Image.Resampling.LANCZOS,
        )
        edited_image_array = np.array(edited_image_resized)
    if mask2 is None:
        mask2 = mask1
    if background:
        input_image_masked = (
            flatten_and_select(input_image_array, ~mask1, channels=channels)
            / 255
        )
        edited_image_masked = (
            flatten_and_select(edited_image_array, ~mask2, channels=channels)
            / 255
        )
    else:
        input_image_masked = (
            flatten_and_select(input_image_array, mask1, channels=channels)
            / 255
        )  # normalize
        edited_image_masked = (
            flatten_and_select(edited_image_array, mask2, channels=channels)
            / 255
        )

    return float(np.mean((input_image_masked - edited_image_masked) ** 2))


def get_normalized_background_score(number: float) -> float:
    if number is np.nan:
        return 0.0
    return np.clip((number - 0.9) * 10, 0, 1)  # type: ignore[no-any-return]


def unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    # type: ignore[no-any-return]
    return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))).item()


def mask_iou(mask_1: npt.NDArray, mask_2: npt.NDArray) -> float:
    intersection = (mask_1 * mask_2).sum()

    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_1, mask_2).astype(np.uint8).sum()

    return float(intersection / union)


def pad_into_shape_2d(
    array: npt.NDArray,
    shape: tuple[int, ...],
) -> npt.NDArray:
    assert len(array.shape) == len(shape) == 2  # noqa: PLR2004, S101

    resized = np.zeros(shape)
    resized[: array.shape[0], : array.shape[1]] = array

    return resized


def translate_to_top_left_2d(array: npt.NDArray) -> npt.NDArray:
    assert len(array.shape) == 2  # noqa: PLR2004, S101

    y_nonzero_indices, x_nonzero_indices = np.nonzero(array)
    min_y = np.min(y_nonzero_indices)
    min_x = np.min(x_nonzero_indices)

    translated = np.zeros_like(array)
    translated[y_nonzero_indices - min_y, x_nonzero_indices - min_x] = array[
        y_nonzero_indices,
        x_nonzero_indices,
    ]

    return translated


def aligned_mask_iou(mask_1: npt.NDArray, mask_2: npt.NDArray) -> float:
    assert mask_1.shape == mask_2.shape  # noqa: S101

    aligned_mask_1 = translate_to_top_left_2d(mask_1)
    aligned_mask_2 = translate_to_top_left_2d(mask_2)

    return mask_iou(aligned_mask_1, aligned_mask_2)


def compute_normalized_rgb_hist_1d(
    rgb_image: Image.Image,
    mask: npt.NDArray[np.bool_] | None = None,
    num_bins: int = 256,
) -> npt.NDArray[np.uint]:
    cv_rgb_image = np.array(rgb_image)

    bgr_histograms = [
        cv2.calcHist(
            [cv_rgb_image],
            [color],
            mask.astype(np.uint8) if mask is not None else None,
            [num_bins],
            [0, 256],
        )
        for color in range(3)  # RED = 0, GREEN = 1, BLUE = 2
    ]

    normalized_bgr_histograms = [
        cv2.normalize(hist, hist) for hist in bgr_histograms
    ]

    # type: ignore[no-any-return]
    return np.concatenate(normalized_bgr_histograms, axis=0).reshape(-1)


def compute_normalized_rgb_hist_3d(
    rgb_image: Image.Image,
    mask: npt.NDArray[np.bool_] | None = None,
    num_bins: int = 8,
) -> npt.NDArray[np.uint]:
    cv_rgb_image = np.array(rgb_image)

    color_hist = cv2.calcHist(
        [cv_rgb_image],
        [0, 1, 2],  # RED = 0, BLUE = 1, GREEN = 2
        mask.astype(np.uint8) if mask is not None else None,
        [num_bins, num_bins, num_bins],
        [0, 256, 0, 256, 0, 256],
    )

    return cv2.normalize(color_hist, color_hist)  # type: ignore[no-any-return]


class HistogramComparisonMethod(enum.IntEnum):
    CORRELATION = cv2.HISTCMP_CORREL
    INTERSECTION = cv2.HISTCMP_INTERSECT
    CHI_SQ = cv2.HISTCMP_CHISQR
    CHI_SQ_ALT = cv2.HISTCMP_CHISQR_ALT
    HELLINGER = cv2.HISTCMP_HELLINGER
    BHATTACHARYYA = cv2.HISTCMP_BHATTACHARYYA
    KL_DIVERGENCE = cv2.HISTCMP_KL_DIV


def compare_color_histograms(  # noqa: PLR0913
    img_1: Image.Image,
    mask_1: npt.NDArray[np.bool_],
    img_2: Image.Image,
    mask_2: npt.NDArray[np.bool_],
    method: HistogramComparisonMethod = HistogramComparisonMethod.CORRELATION,
    num_bins: int = 32,
    smoothing_sigma: float = 5,
) -> tuple[float, Image.Image]:
    hist_1 = compute_normalized_rgb_hist_1d(
        img_1,
        mask=mask_1,
        num_bins=num_bins,
    )
    hist_2 = compute_normalized_rgb_hist_1d(
        img_2,
        mask=mask_2,
        num_bins=num_bins,
    )

    channels = 3
    smooth_hist_1 = np.concatenate(
        [
            gaussian_filter1d(channel_hist, smoothing_sigma, mode="nearest")
            for channel_hist in np.split(hist_1, channels)
        ],
    )
    smooth_hist_2 = np.concatenate(
        [
            gaussian_filter1d(channel_hist, smoothing_sigma, mode="nearest")
            for channel_hist in np.split(hist_2, channels)
        ],
    )

    similarities = [
        cv2.compareHist(channel_hist_1, channel_hist_2, method)
        for channel_hist_1, channel_hist_2 in zip(
            np.split(smooth_hist_1, channels),
            np.split(smooth_hist_2, channels),
            strict=True,
        )
    ]

    score = np.mean(similarities)

    color_histogram_figure = plotting.plot_rgb_histograms(
        np.stack([smooth_hist_1, smooth_hist_2]),
    )

    return score, figure_to_image(color_histogram_figure)
