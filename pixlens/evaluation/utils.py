from collections import Counter
import logging
import colorspacious as cs
import numpy as np
import torch
from PIL import Image, ImageColor

# import delta e color similarity function
from skimage.color import deltaE_ciede2000, rgb2lab

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
    # cropped_array = image_array * boolean_mask[:, :, np.newaxis]
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
    # assert Counter(colors) == Counter(colorsn _pil_method)

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

    logging.debug("Dominant color: %s", dominant_color)
    logging.debug("Closest colors: %s", closest_colors)

    # Check if the target color is in the 5 closest colors
    return target_color in [color[0] for color in closest_colors]
