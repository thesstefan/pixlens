from collections import Counter

import numpy as np
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
]
edits = list(EditType)
new_object = ["object_addition", "object_replacement", "background", "texture"]
new_object_with_indication = ["alter_parts", "positional_addition"]
same_object = [
    edit
    for edit in edits
    if edit not in new_object + new_object_with_indication
]
tol = 1e-6
DIVDING_BY_ZERO_MSG = "Cannot divide by zero"


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
    if area2 < tol:
        raise ValueError(DIVDING_BY_ZERO_MSG)
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
    opposite: bool = False,
) -> NDArray:
    # Ensure the mask is a boolean array
    mask = mask.astype(bool)
    if mask.shape != np_image.shape[:2]:
        mask = mask.T  # transpose mask

    # Apply the mask to each channel
    masked_image = np.zeros_like(np_image)
    if opposite:
        # Set masked areas to white
        for i in range(
            np_image.shape[2],
        ):  # Assuming image has shape [Height, Width, Channels]
            masked_image[:, :, i] = np.where(~mask, 255, np_image[:, :, i])
    else:
        for i in range(
            np_image.shape[2],
        ):  # Assuming image has shape [Height, Width, Channels]
            masked_image[:, :, i] = np_image[:, :, i] * mask

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
            opposite=True,
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


def compute_union_segmentation_masks(masks: list[NDArray]) -> NDArray:
    if not masks:
        raise ValueError("The list of masks cannot be empty")

    union_mask = masks[0]  # First type mustmask be from image 1.
    for mask in masks[1:]:
        if mask.shape != union_mask.shape:
            resized_mask = np.zeros_like(union_mask, dtype=bool)
            resized_mask[: mask.shape[0], : mask.shape[1]] = mask
        else:
            resized_mask = mask

        union_mask = np.bitwise_or(union_mask, resized_mask)

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


def flatten_and_select(np_image: NDArray, mask: NDArray) -> NDArray:
    mask = mask.astype(bool)
    # Flatten both image and mask
    flat_image: NDArray = np_image.reshape(-1, np_image.shape[-1])
    flat_mask = mask.flatten()
    return flat_image[flat_mask]


def compute_mse_over_mask(
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
    input_image_masked = (
        flatten_and_select(input_image_array, mask1) / 255
    )  # normalize
    edited_image_masked = flatten_and_select(edited_image_array, mask2) / 255
    if background:
        input_image_masked = (
            flatten_and_select(
                input_image_array,
                ~mask1,
            )
            / 255
        )
        edited_image_masked = (
            flatten_and_select(edited_image_array, ~mask2) / 255
        )
    return float(np.mean((input_image_masked - edited_image_masked) ** 2))


def extract_decimal_part(number: float) -> float:
    if number < 0.9:
        return 0.0
    # Convert the number to a string to work with its decimal part
    str_number = str(number)

    # Find the position of the decimal point
    decimal_point_index = str_number.find(".")

    # Extract the decimal part from the second decimal place onwards
    if decimal_point_index != -1:
        return float("0." + str_number[decimal_point_index + 2 :])
    return 0.0
