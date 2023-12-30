import logging
from pathlib import Path

import pandas as pd

from pixlens.evaluation.interfaces import Edit, EditType

change_action_dict = {
    "sit": "sitting",
    "run": "running",
    "hit": "hitting",
    "jump": "jumping",
    "stand": "standing",
    "lay": "laying",
}


def log_model_if_not_in_cache(model_name: str, cache_dir: Path) -> None:
    model_dir = model_name.replace("/", "--")
    model_dir = "models--" + model_dir
    full_path = cache_dir / model_dir
    if not full_path.is_dir():
        logging.info(
            "Downloading model from %s ...",
            model_name,
        )


def generate_description_based_prompt(edit: Edit) -> str:
    edit_type = edit.edit_type
    from_attribute = edit.from_attribute
    to_attribute = edit.to_attribute
    category = edit.category

    prompt_templates: dict[EditType, str] = {
        # TODO: add prompts in a description based way
        # e.g. instead of "Add a red apple to the image", "A red apple".
        EditType.COLOR: f"A photo of a {category}[SEP]A photo of a {to_attribute} {category}",  # noqa: E501
        EditType.SIZE: f"A photo of a {category}[SEP]A photo of a {to_attribute} {category}",  # noqa: E501
        EditType.OBJECT_ADDITION: f"A photo of a {category}[SEP]A photo of a {category} and a {to_attribute}",  # noqa: E501
        EditType.POSITIONAL_ADDITION: f"A photo of a {category}[SEP]A photo of a {category} and a {to_attribute}",  # noqa: E501
        EditType.OBJECT_REMOVAL: f"A photo of a {category}[SEP]A photo without a {category}",  # noqa: E501 FIXME: this one is HARD!
        EditType.OBJECT_REPLACEMENT: f"A photo of a {from_attribute}[SEP]A photo of a {to_attribute}",  # noqa: E501
        EditType.POSITION_REPLACEMENT: f"A photo of a {category} on the {from_attribute}[SEP]A photo of a {category} on the {to_attribute}",  # noqa: E501
        EditType.OBJECT_DUPLICATION: f"A photo of a {category}[SEP]A photo of two {category}s",  # noqa: E501
        EditType.TEXTURE: f"A photo of a {category}[SEP]A photo of a {category} with {to_attribute} texture",  # noqa: E501
        EditType.ACTION: f"A photo of a {category} {change_action_dict[from_attribute]}[SEP]A photo of a {category} {change_action_dict[to_attribute]}"  # noqa: E501
        if from_attribute is not None and not pd.isna(from_attribute)
        else f"A photo of a {category}[SEP]A photo of a {category} {change_action_dict[to_attribute]}",  # noqa: E501
        EditType.VIEWPOINT: f"A photo of a {category} from {from_attribute}[SEP]A photo of a {category} from {to_attribute}"  # noqa: E501
        if from_attribute is not None and not pd.isna(from_attribute)
        else f"A photo of a {category}[SEP]A photo of a {category} from {to_attribute}",  # noqa: E501
        # also it could be (according to EDITVAL):
        # EditType.VIEWPOINT: f"A photo of a {from_attribute} view of a {category} [SEP]A photo of a {to_attribute} view of a {category}"  # noqa: E501, ERA001
        EditType.BACKGROUND: f"A photo of a {category} in the {from_attribute}[SEP]A photo of a {category} in the {to_attribute}"  # noqa: E501
        if from_attribute is not None and not pd.isna(from_attribute)
        else f"A photo of a {category}[SEP]A photo of a {category} in the {to_attribute}",  # noqa: E501
        EditType.STYLE: f"A photo of a {category}[SEP]An art painting of a {category} in {to_attribute} style",  # noqa: E501
        EditType.SHAPE: f"A photo of a {category}[SEP]A photo of a {category} in the shape of a {to_attribute}",  # noqa: E501
        EditType.ALTER_PARTS: f"A photo of a {category}[SEP]{to_attribute} to the {category}",  # noqa: E501
        # THIS IS ABSOLUTE BS. Fuck you EDITVAL. for "alter parts" there is no way to write a description based prompt  # noqa: E501
        # as basically "to_attribute" is always something like: "add tomato toppings to...", which is instruction based.  # noqa: E501
    }

    if edit_type in prompt_templates:
        return prompt_templates[edit_type]
    error_msg = f"Unknown edit type: {edit_type}"
    raise ValueError(error_msg)


def generate_instruction_based_prompt(edit: Edit) -> str:
    edit_type = edit.edit_type
    from_attribute = edit.from_attribute
    to_attribute = edit.to_attribute
    category = edit.category

    prompt_templates = {
        EditType.COLOR: f"Change the color of the {category} to {to_attribute}",
        EditType.SIZE: f"Change the size of the {category} to {to_attribute}",
        EditType.OBJECT_ADDITION: f"Add a {to_attribute} to the image",
        EditType.POSITIONAL_ADDITION: f"Add a {to_attribute} the {category}",
        EditType.OBJECT_REMOVAL: f"Remove the {category}",
        EditType.OBJECT_REPLACEMENT: f"Replace the {from_attribute} with a {to_attribute}",  # noqa: E501
        EditType.POSITION_REPLACEMENT: f"Move the {category} from the {from_attribute} to the {to_attribute} of the picture in the image",  # noqa: E501
        EditType.OBJECT_DUPLICATION: f"Duplicate the {category}",
        EditType.TEXTURE: f"Change the texture of the {category} to {to_attribute}",  # noqa: E501
        EditType.ACTION: f"Transform the {category}'s pose from {from_attribute} to {to_attribute}"  # noqa: E501
        if from_attribute is not None and not pd.isna(from_attribute)
        else f"Transform the {category}'s pose to {to_attribute}",
        EditType.VIEWPOINT: f"Change the viewpoint of the {category} from {from_attribute} to {to_attribute}"  # noqa: E501
        if from_attribute is not None and not pd.isna(from_attribute)
        else f"Change the viewpoint of the {category} to {to_attribute}",
        EditType.BACKGROUND: f"Change the background from {from_attribute} to {to_attribute}"  # noqa: E501
        if from_attribute is not None and not pd.isna(from_attribute)
        else f"Change the background to {to_attribute}",
        EditType.STYLE: f"Change the style of the {category} to {to_attribute}",
        EditType.SHAPE: f"Change the shape of the {category} to {to_attribute}",
        EditType.ALTER_PARTS: f"{to_attribute} to the {category}",
    }

    if edit_type in prompt_templates:
        return prompt_templates[edit_type]
    error_msg = f"Unknown edit type: {edit_type}"
    raise ValueError(error_msg)
