import logging
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageColor

# import delta e color similarity function
from skimage.color import deltaE_cie76, rgb2lab

from pixlens.evaluation.interfaces import Edit, EditType

directions_and_instructions = ["add", "to", "right", "left", "below"]
edits = list(EditType)
new_object = ["object_addition", "object_replacement", "background", "texture"]
new_object_with_indication = ["alter_parts", "positional_addition"]
same_object = [
    edit.type_name
    for edit in edits
    if edit.type_name not in new_object + new_object_with_indication
]
tol = 1e-6
SHAPE_DIFFERENCE_MSG = "Input and output shapes must be the same shape"
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


def get_updated_to(edit: Edit) -> str | None:
    if edit.edit_type.type_name in new_object:
        return edit.to_attribute
    if edit.edit_id in new_object_with_indication:
        return remove_words_from_string(
            edit.to_attribute,
            directions_and_instructions,
        )
    return None


def get_prompt_for_output_detection(edit: Edit) -> str:
    if edit.edit_type.type_name in new_object:
        return edit.to_attribute
    if edit.edit_id in new_object_with_indication:
        return remove_words_from_string(
            edit.to_attribute,
            directions_and_instructions,
        )
    return edit.category


def get_prompt_for_input_detection(edit: Edit) -> str:
    if edit.edit_type.type_name in ("background", "object_replacement"):
        return edit.from_attribute
    return edit.category


def compute_area(tensor1: torch.Tensor) -> float:
    area1 = torch.sum(tensor1)
    return area1.item()


def compute_area_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> float:
    if numerator.shape != denominator.shape:
        raise ValueError(SHAPE_DIFFERENCE_MSG)
    area1 = compute_area(numerator)
    area2 = compute_area(denominator)
    if area2 < tol:
        raise ValueError(DIVDING_BY_ZERO_MSG)
    return area1 / area2


def compute_iou(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    if tensor1.shape != tensor2.shape:
        raise ValueError(SHAPE_DIFFERENCE_MSG)
    intersection = torch.logical_and(tensor1, tensor2).sum()
    union = torch.logical_or(tensor1, tensor2).sum()
    iou = intersection.float() / union.float()
    return iou.item()


def is_small_area_within_big_area(
    input_mask: torch.Tensor,
    edited_mask: torch.Tensor,
    confidence_threshold: float = 0.9,
) -> bool:
    # infer small and big area from input and edited masks
    if compute_area(input_mask) > compute_area(edited_mask):
        small_area = edited_mask
        big_area = input_mask
    else:
        small_area = input_mask
        big_area = edited_mask

    if small_area.shape != big_area.shape:
        raise ValueError(SHAPE_DIFFERENCE_MSG)
    intersection = torch.logical_and(small_area, big_area).sum()
    return intersection.item() / compute_area(small_area) > confidence_threshold


def get_colors_in_masked_area(
    image: Image.Image,
    mask: torch.Tensor,
) -> list[tuple[int, int, int]]:
    image_array = np.array(image)

    boolean_mask = mask[0].cpu().numpy()
    cropped_array = image_array[boolean_mask]

    # show image from cropped array
    logging.debug("Masked part in edited image:")
    Image.fromarray(image_array * boolean_mask[:, :, np.newaxis]).show()

    pixels = cropped_array.reshape(-1, cropped_array.shape[-1])

    # Convert the 2D array to a list of tuples
    pixel_tuples = [tuple(pixel) for pixel in pixels]

    # Use Counter to count occurrences of each unique color tuple
    color_counts = Counter(pixel_tuples)

    # Convert the Counter items to an array of tuples (count, color)
    # and then sort in descending order by count
    color_counts = sorted(
        color_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(count, color) for color, count in color_counts]


def calculate_h_index(sorted_counts: list[int]) -> int:
    return next(
        (i for i, count in enumerate(sorted_counts, start=1) if count < i),
        len(sorted_counts),
    )


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
    # ensure that the image is in rgb format
    image = image.convert("RGB")

    # Get the colors in the masked area
    colors = get_colors_in_masked_area(image, mask)

    # get h.index of colors in masked area
    h_index = calculate_h_index([count for count, _ in colors])

    # Get the most dominant colors based on the H-index
    dominant_colors = [(count, rgb2lab(color)) for count, color in colors[:100]]

    # colors_pil_method = image.getcolors(image.size[0] * image.size[1])
    # assert that colors and colors_pil_method contain the same keys and values (order doesn't matter)
    # assert Counter(colors) == Counter(colorsn _pil_method)

    # Get the most common color
    # dominant_color = rgb2lab(max(colors, key=lambda x: x[0])[1])

    # compute pairwise distance between colors in PIL.ImageColor.colormap and
    # colors detected in the masked area. The distance should be weighted by
    # the number of pixels that have that color in the masked area, then
    # take the 5 colors in PIL.ImageColor.colormap that have the smallest
    # average distance to the colors in the masked area

    # sum all counts in colors
    total_count = sum([count for count, _ in dominant_colors])

    # compute the average distance between each color in PIL.ImageColor.colormap
    # and the colors in the masked area
    average_distances = []
    for pil_color in ImageColor.colormap.items():
        color_lab = rgb2lab(
            ImageColor.getrgb(pil_color[1])
            if not isinstance(pil_color[1], tuple)
            else pil_color[1]
        )
        average_distance = 0.0
        for count, color in dominant_colors:
            average_distance += count * deltaE_cie76(color_lab, color)
        average_distances.append((average_distance / total_count, pil_color))

    average_distances.sort(key=lambda x: x[0])

    # get the 5 colors in PIL.ImageColor.colormap that have the smallest
    # average distance to the colors in the masked area
    closest_colors = average_distances[:10]

    logging.info("Closest colors: %s", closest_colors)

    # Check if the target color is in the 5 closest colors
    return target_color in [color[0] for color in closest_colors]
