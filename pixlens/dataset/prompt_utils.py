import pandas as pd

from pixlens.evaluation.interfaces import EditType, Edit

PROMPT_SEP = "[SEP]"


change_action_dict = {
    "sit": "sitting",
    "run": "running",
    "hit": "hitting",
    "jump": "jumping",
    "stand": "standing",
    "lay": "laying",
}


def generate_description_based_prompt(
    edit_type: EditType,
    from_attribute: str | None,
    to_attribute: str | None,
    category: str,
) -> str:
    category = "".join(
        char if char.isalpha() or char.isspace() else " " for char in category
    )

    if to_attribute:
        to_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in to_attribute or ""
        )

    if from_attribute:
        from_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in from_attribute or ""
        )

    prompt_templates: dict[EditType, str] = {
        # TODO: Remove redundant "A photo of" prefix
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
            f"A photo of {from_attribute} {category}s"
            f"{PROMPT_SEP}A photo of no {category}s"
        ),
        EditType.SINGLE_INSTANCE_REMOVAL: (
            f"A photo of a {category}"
            f"{PROMPT_SEP}A photo of a {category} with one less {category}"
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
            if to_attribute in change_action_dict
            and from_attribute in change_action_dict
            else (
                f"A photo of a {category}"
                f"{PROMPT_SEP}A photo of a "
                f"{category} {change_action_dict[to_attribute]}"
                if to_attribute in change_action_dict
                else ""
            )
        ),
        EditType.VIEWPOINT: (
            f"A photo of a {category} from {from_attribute}"
            f"{PROMPT_SEP}A photo of a {category} from {to_attribute}"
            if from_attribute is not None and not pd.isna(from_attribute)
            else (
                f"A photo of a {category}"
                f"{PROMPT_SEP}A photo of a {category} from {to_attribute}"
            )
        ),
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
            f"A photo of a {category}"
            f"{PROMPT_SEP}{to_attribute} to the {category}"
        ),
    }

    if edit_type in prompt_templates:
        return prompt_templates[edit_type]
    error_msg = f"Unknown edit type: {edit_type}"
    raise ValueError(error_msg)


def generate_simplified_description_based_prompt(
    edit_type: EditType,
    from_attribute: str | None,
    to_attribute: str | None,
    category: str,
) -> str:
    category = "".join(
        char if char.isalpha() or char.isspace() else " " for char in category
    )

    if not pd.isna(to_attribute):
        to_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in to_attribute
        )

    if not pd.isna(from_attribute):
        from_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in from_attribute
        )

    prompt_templates: dict[EditType, str] = {
        EditType.COLOR: f"{to_attribute} {category}",
        EditType.SIZE: f"{to_attribute} {category}",
        EditType.OBJECT_ADDITION: f"{category} and a {to_attribute}",
        EditType.POSITIONAL_ADDITION: f"{category} and a {to_attribute}",
        EditType.OBJECT_REMOVAL: f"Remove {category}",
        EditType.OBJECT_REPLACEMENT: f"{to_attribute}",
        EditType.POSITION_REPLACEMENT: f"{category} on the {to_attribute}",
        EditType.OBJECT_DUPLICATION: f"two {category}s",
        EditType.TEXTURE: f"{category} with {to_attribute} texture",
        EditType.ACTION: (
            f"{category} {change_action_dict[to_attribute]}"
            if to_attribute in change_action_dict
            else ""
        ),
        EditType.VIEWPOINT: f"{category} from {to_attribute}",
        EditType.BACKGROUND: f"{category} in the {to_attribute}",
        EditType.STYLE: f"art painting of a {category} in {to_attribute} style",
        EditType.SHAPE: f"{category} in the shape of a {to_attribute}",
        EditType.ALTER_PARTS: f"{to_attribute} to the {category}",
    }

    if edit_type in prompt_templates:
        return prompt_templates[edit_type]
    error_msg = f"Unknown edit type: {edit_type}"
    raise ValueError(error_msg)


def generate_original_description(edit: Edit) -> str:
    edit_type = edit.edit_type
    from_attribute = edit.from_attribute
    category = edit.category

    category = "".join(
        char if char.isalpha() or char.isspace() else " "
        for char in edit.category
    )

    if not pd.isna(edit.from_attribute):
        from_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in edit.from_attribute
        )
    if edit_type == EditType.OBJECT_REPLACEMENT:
        return from_attribute
    if edit_type == EditType.BACKGROUND:
        if from_attribute is not None and not pd.isna(from_attribute):
            return f"{from_attribute} background"
        return "background"
    if edit_type == EditType.ACTION:
        if not pd.isna(from_attribute):
            return f"{category} {change_action_dict[from_attribute]}"
        return category
    if edit_type == EditType.VIEWPOINT:
        return f"{category} from {from_attribute}"
    if edit_type == EditType.POSITION_REPLACEMENT:
        return f"{category} on the {from_attribute}"
    return category


def generate_instruction_based_prompt(
    edit_type: EditType,
    from_attribute: str | None,
    to_attribute: str | None,
    category: str,
) -> str:
    category = "".join(
        char if char.isalpha() or char.isspace() else " " for char in category
    )

    if not pd.isna(to_attribute):
        to_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in to_attribute or ""
        )

    if not pd.isna(from_attribute):
        from_attribute = "".join(
            char if char.isalpha() or char.isspace() else " "
            for char in from_attribute or ""
        )

    prompt_templates = {
        EditType.COLOR: f"Change the color of the {category} to {to_attribute}",
        EditType.SIZE: f"Change the size of the {category} to {to_attribute}",
        EditType.OBJECT_ADDITION: f"Add a {to_attribute} to the image",
        EditType.POSITIONAL_ADDITION: f"Add a {to_attribute} of the {category}",
        EditType.OBJECT_REMOVAL: f"Remove the {category}",
        EditType.SINGLE_INSTANCE_REMOVAL: f"Remove just one {category}",
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
