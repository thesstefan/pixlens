import torch

from pixlens.evaluation.interfaces import Edit, EditType

directions_and_instructions = ["add", "to", "right", "left", "below"]
edits = list(EditType)
new_object = ["object_addition", "object_replacement", "background"]
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


def get_prompt_for_output_detection(edit: Edit) -> str:
    if edit.edit_type in new_object:
        return edit.to_attribute
    if edit.edit_id in new_object_with_indication:
        return remove_words_from_string(
            edit.to_attribute, directions_and_instructions
        )
    return edit.category


def get_prompt_for_input_detection(edit: Edit) -> str:
    if edit.edit_type == "background":
        return edit.from_attribute
    return edit.category


def compute_area(tensor1: torch.Tensor) -> float:
    area1 = torch.sum(tensor1)
    return area1.item()


def compute_area_ratio(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    area1 = compute_area(tensor1)
    area2 = compute_area(tensor2)
    if area2:
        return area1 / area2
    msg = "Area of tensor2 is 0"
    raise ValueError(msg)


def compute_iou(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    intersection = torch.logical_and(tensor1, tensor2).sum()
    union = torch.logical_or(tensor1, tensor2).sum()
    iou = intersection.float() / union.float()
    return iou.item()
