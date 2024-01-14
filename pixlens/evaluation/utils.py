from collections import Counter

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from PIL import Image, ImageColor

# import delta e color similarity function
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim

from pixlens.evaluation.interfaces import Edit, EditType, EvaluationInput

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


def compute_area(mask: npt.NDArray[np.bool_]) -> float:
    return float(np.sum(mask))


def compute_area_ratio(
    numerator: npt.NDArray[np.bool_],
    denominator: npt.NDArray[np.bool_],
) -> float:
    area1 = compute_area(numerator)
    area2 = compute_area(denominator)
    if not area2 > 0:
        raise ZeroDivisionError
    return area1 / area2


def is_small_area_within_big_area(
    input_mask: npt.NDArray[np.bool_],
    edited_mask: npt.NDArray[np.bool_],
    confidence_threshold: float = 0.9,
) -> bool:
    if compute_area(input_mask) > compute_area(edited_mask):
        small_area = edited_mask
        big_area = input_mask
    else:
        small_area = input_mask
        big_area = edited_mask
    intersection = np.logical_and(small_area, big_area).sum()
    result: bool = (
        intersection / compute_area(small_area)
    ) > confidence_threshold
    return result


def get_colors_in_masked_area(
    image: Image.Image,
    mask: torch.Tensor,
) -> list[tuple[int, int, int]]:
    image_array = np.array(image)

    boolean_mask = mask[0].cpu().numpy()
    # cropped_array = image_array * boolean_mask[:, :, np.newaxis]
    cropped_array = image_array[boolean_mask]

    pixels = cropped_array.reshape(-1, cropped_array.shape[-1])

    # Convert the 2D array to a list of tuples
    pixel_tuples = [tuple(pixel) for pixel in pixels]

    # Use Counter to count occurrences of each unique color tuple
    color_counts = Counter(pixel_tuples)

    # Convert the Counter items to an array of tuples (count, color)
    return [(count, color) for color, count in color_counts.items()]


# TODO: this function is clearly limited and should not be used in the final
# definitive version. Imagine the target color is "ocean blue"
# and the 5 closest colors are "blue", "light blue", "dark blue", "sky blue",
# "blueberry". The function will return False because "ocean blue" is not in
# the 5 closest colors. However, the edit is still correct.
def color_change_applied_correctly(
    image: Image.Image,
    mask: torch.Tensor,
    target_color: str,
) -> bool:
    # Get the colors in the masked area
    colors = get_colors_in_masked_area(image, mask)

    # colors_pil_method = image.getcolors(image.size[0] * image.size[1])
    # assert that colors and colors_pil_method contain the same keys and values (order doesn't matter)
    # assert Counter(colors) == Counter(colors_pil_method)

    # Get the most common color
    dominant_color = rgb2lab(max(colors, key=lambda x: x[0])[1])

    # Find the 5 colors in PIL.ImageColor.colormap that are most similar
    # to the retrieved color based on Delta-E distance
    # however the colors in the colormap are in hexadecimal and the dominant
    # color is in RGB, so we need to convert all colors to Lab space as it is
    # required by the delta_e_cie2000 function
    closest_colors = sorted(
        ImageColor.colormap.items(),
        key=lambda color: deltaE_ciede2000(
            rgb2lab(ImageColor.getrgb(color[1])),
            dominant_color,
        ),
    )[:10]

    # Check if the target color is in the 5 closest colors
    return target_color in [color[0] for color in closest_colors]


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
    return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))).item()  # type: ignore[no-any-return]


def compute_color_hist_vector(
    image: Image.Image,
    mask: npt.NDArray[np.bool_] | None = None,
    bins: int = 32,
) -> npt.NDArray[np.uint]:
    cv_bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    bgr_hist: npt.NDArray[np.uint] = np.concatenate(
        [
            cv2.calcHist(
                [cv_bgr_image],
                [color],
                mask.astype(np.uint8) if mask is not None else None,
                [bins],
                [0, 256],
            )
            for color in range(3)  # BLUE = 0, GREEN = 1, RED = 2
        ],
        axis=0,
    )

    return bgr_hist.reshape(-1)


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
