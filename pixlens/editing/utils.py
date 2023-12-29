import logging
from pathlib import Path

import numpy as np

from pixlens.evaluation.interfaces import Edit, EditType


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

    prompt_templates = {
        # TODO: add prompts in a description based way
        # e.g. instead of "Add a red apple to the image", "A red apple".
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
        EditType.OBJECT_ADDITION: f"Add a {to_attribute} to the {category}",
        EditType.POSITIONAL_ADDITION: f"Add a {to_attribute} the {category}",
        EditType.OBJECT_REMOVAL: f"Remove the {category}",
        EditType.OBJECT_REPLACEMENT: f"Replace the {from_attribute} with a {to_attribute}",  # noqa: E501
        EditType.POSITION_REPLACEMENT: f"Move the {category} from the {from_attribute} to the {to_attribute} of the picture in the image",  # noqa: E501
        EditType.OBJECT_DUPLICATION: f"Duplicate the {category}",
        EditType.TEXTURE: f"Change the texture of the {category} to {to_attribute}",  # noqa: E501
        EditType.ACTION: f"Transform the {category}'s pose from {from_attribute} to {to_attribute}"  # noqa: E501
        if from_attribute is not None and not np.isnan(from_attribute)
        else f"Transform the {category}'s pose to {to_attribute}",
        EditType.VIEWPOINT: f"Change the viewpoint of the {category} from {from_attribute} to {to_attribute}"  # noqa: E501
        if from_attribute is not None and not np.isnan(from_attribute)
        else f"Change the viewpoint of the {category} to {to_attribute}",
        EditType.BACKGROUND: f"Change the background from {from_attribute} to {to_attribute}"  # noqa: E501
        if from_attribute is not None and not np.isnan(from_attribute)
        else f"Change the background to {to_attribute}",
        EditType.STYLE: f"Change the style of the {category} to {to_attribute}",
        EditType.SHAPE: f"Change the shape of the {category} to {to_attribute}",
        EditType.ALTER_PARTS: f"{to_attribute} to the {category}",
    }

    if edit_type in prompt_templates:
        return prompt_templates[edit_type]
    error_msg = f"Unknown edit type: {edit_type}"
    raise ValueError(error_msg)
