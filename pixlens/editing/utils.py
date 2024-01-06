import logging
from pathlib import Path

import pandas as pd

from pixlens.evaluation.interfaces import Edit, EditType

PROMPT_SEP = "[SEP]"

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

    category = "".join(
        char if char.isalpha() or char.isspace() else " "
        for char in edit.category
    )

    if not pd.isna(edit.to_attribute):
        to_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in edit.to_attribute
        )

    if not pd.isna(edit.from_attribute):
        from_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in edit.from_attribute
        )

    prompt_templates: dict[EditType, str] = {
        # TODO: add prompts in a description based way
        # e.g. instead of "Add a red apple to the image", "A red apple".
        EditType.COLOR: (
            f"A photo of a {category}"
            f"{PROMPT_SEP}A photo of a {to_attribute} {category}"
        ),
        EditType.SIZE: (
            f"A photo of a {category}"
            f"{PROMPT_SEP}A photo of a {to_attribute} {category}"
        ),
        EditType.OBJECT_ADDITION: (
            f"A photo of a {category}"
            f"{PROMPT_SEP}A photo of a {category} and a {to_attribute}"
        ),
        EditType.POSITIONAL_ADDITION: (
            f"A photo of a {category}"
            f"{PROMPT_SEP}A photo of a {category} and a {to_attribute}"
        ),
        EditType.OBJECT_REMOVAL: (
            # FIXME: this one is HARD!
            f"A photo of a {category}"
            f"{PROMPT_SEP}A photo without a {category}"
        ),
        EditType.OBJECT_REPLACEMENT: (
            f"A photo of a {from_attribute}"
            f"{PROMPT_SEP}A photo of a {to_attribute}"
        ),
        EditType.POSITION_REPLACEMENT: (
            f"A photo of a {category} on the {from_attribute}"
            f"{PROMPT_SEP}A photo of a {category} on the {to_attribute}"
        ),
        EditType.OBJECT_DUPLICATION: (
            f"A photo of a {category}{PROMPT_SEP}A photo of two {category}s"
        ),
        EditType.TEXTURE: (
            f"A photo of a {category}"
            f"{PROMPT_SEP}A photo of a {category} with {to_attribute} texture"
        ),
        EditType.ACTION: (
            f"A photo of a {category} {change_action_dict[from_attribute]}"
            f"{PROMPT_SEP}A photo of a "
            f"{category} {change_action_dict[to_attribute]}"
            if from_attribute is not None and not pd.isna(from_attribute)
            else (
                f"A photo of a {category}"
                f"{PROMPT_SEP}A photo of a "
                f"{category} {change_action_dict[to_attribute]}"
            )
        ),
        EditType.VIEWPOINT: (
            f"A photo of a {category} from {from_attribute}"
            "{PROMPT_SEP}A photo of a {category} from {to_attribute}"
            if from_attribute is not None and not pd.isna(from_attribute)
            else (
                f"A photo of a {category}"
                f"{PROMPT_SEP}A photo of a {category} from {to_attribute}"
            )
        ),
        # also it could be (according to EDITVAL):
        # EditType.VIEWPOINT: (
        #    f"A photo of a {from_attribute} view of a {category}"
        #    f"{PROMPT_SEP}A photo of a {to_attribute} view of a {category}"
        # )
        EditType.BACKGROUND: (
            f"A photo of a {category} in the {from_attribute}"
            f"{PROMPT_SEP}A photo of a {category} in the {to_attribute}"
            if from_attribute is not None and not pd.isna(from_attribute)
            else (
                f"A photo of a {category}"
                f"{PROMPT_SEP}A photo of a {category} in the {to_attribute}"
            )
        ),
        EditType.STYLE: (
            f"A photo of a {category}"
            f"{PROMPT_SEP}An art painting of a "
            f"{category} in {to_attribute} style"
        ),
        EditType.SHAPE: (
            f"A photo of a {category}"
            f"{PROMPT_SEP}A photo of a "
            f"{category} in the shape of a {to_attribute}"
        ),
        EditType.ALTER_PARTS: (
            # THIS IS ABSOLUTE BS. Fuck you EDITVAL. for "alter parts" there
            # is no way to write a description based prompt as basically
            # "to_attribute" is always something like:
            # "add tomato toppings to...", which is instruction based.
            f"A photo of a {category}"
            f"{PROMPT_SEP}{to_attribute} to the {category}"
        ),
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

    category = "".join(
        char if char.isalpha() or char.isspace() else " "
        for char in edit.category
    )

    if not pd.isna(edit.to_attribute):
        to_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in edit.to_attribute
        )

    if not pd.isna(edit.from_attribute):
        from_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in edit.from_attribute
        )

    prompt_templates = {
        EditType.COLOR: f"Change the color of the {category} to {to_attribute}",
        EditType.SIZE: f"Change the size of the {category} to {to_attribute}",
        EditType.OBJECT_ADDITION: f"Add a {to_attribute} to the image",
        EditType.POSITIONAL_ADDITION: f"Add a {to_attribute} the {category}",
        EditType.OBJECT_REMOVAL: f"Remove the {category}",
        EditType.OBJECT_REPLACEMENT: (
            f"Replace the {from_attribute} with a {to_attribute}"
        ),
        EditType.POSITION_REPLACEMENT: (
            f"Move the {category} from the {from_attribute} to the "
            f"{to_attribute} of the picture in the image"
        ),
        EditType.OBJECT_DUPLICATION: f"Duplicate the {category}",
        EditType.TEXTURE: (
            f"Change the texture of the {category} to {to_attribute}"
        ),
        EditType.ACTION: (
            f"Transform the {category}'s pose from "
            f"{from_attribute} to {to_attribute}"
            if from_attribute is not None and not pd.isna(from_attribute)
            else f"Transform the {category}'s pose to {to_attribute}"
        ),
        EditType.VIEWPOINT: (
            f"Change the viewpoint of the {category} from "
            f"{from_attribute} to {to_attribute}"
            if from_attribute is not None and not pd.isna(from_attribute)
            else f"Change the viewpoint of the {category} to {to_attribute}"
        ),
        EditType.BACKGROUND: (
            f"Change the background from {from_attribute} to {to_attribute}"
            if from_attribute is not None and not pd.isna(from_attribute)
            else f"Change the background to {to_attribute}"
        ),
        EditType.STYLE: f"Change the style of the {category} to {to_attribute}",
        EditType.SHAPE: f"Change the shape of the {category} to {to_attribute}",
        EditType.ALTER_PARTS: f"{to_attribute} to the {category}",
    }

    if edit_type in prompt_templates:
        return prompt_templates[edit_type]
    error_msg = f"Unknown edit type: {edit_type}"
    raise ValueError(error_msg)


def split_description_based_prompt(prompt: str) -> tuple[str, str]:
    src, dst = prompt.split(PROMPT_SEP)

    return src, dst
