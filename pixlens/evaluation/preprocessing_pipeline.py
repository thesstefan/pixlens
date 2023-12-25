import json
import logging
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column

from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation.interfaces import Edit, EditType
from pixlens.utils.utils import get_cache_dir


# create a class that will parse a json object to get some edit instructions
# as a init function it receives the json object file path
# in the init it calls a private function that parses the json object
class PreprocessingPipeline:
    def __init__(self, json_object_path: str, dataset_path: str) -> None:
        self.json_object_path = json_object_path
        self.edit_dataset: pd.DataFrame
        self.dataset_path = dataset_path
        self._parse_json_object()

    def _parse_json_object(self) -> None:
        pandas_path = Path(get_cache_dir(), "edit_dataset.csv")

        if pandas_path.exists():
            self.edit_dataset = pd.read_csv(pandas_path)

            # TODO: add more complex schema validation  # noqa: TD002
            schema = pa.DataFrameSchema(
                {
                    "edit_id": Column(pa.Int),
                    "image_id": Column(pa.Int),
                    "edit_type": Column(pa.String),
                    "class": Column(pa.String),
                    "from_attribute": Column(pa.String, nullable=True),
                    "to_attribute": Column(pa.String),
                    "input_image_path": Column(pa.String),
                },
            )

            # validating the data frame with the expected schema
            try:
                schema.validate(self.edit_dataset, lazy=True)
            except pa.errors.SchemaErrors as err:
                logging.warning("Schema errors and failure cases:")
                logging.warning(err)
                logging.warning(
                    "Deleting cached edit dataset, as it does not comply "
                    "with the established schema",
                )
                pandas_path.unlink()
            else:
                return

        with Path(self.json_object_path).open(encoding="utf-8") as json_file:
            json_data = json.load(json_file)

        records: list[dict] = []

        # Iterate through the JSON data
        for obj_class, images in json_data.items():
            for image_id, edits in images.items():
                for edit_type, values in edits.items():
                    from_values = values.get("from", [""])
                    to_values = values.get("to", [])

                    # Handle case when there is no "from" value
                    if not from_values:
                        from_values = [""] * len(to_values)

                    # Iterate through "to" values
                    for from_val, to_val in zip(
                        from_values,
                        to_values,
                        strict=False,
                    ):
                        records.append(
                            {
                                "edit_id": len(records),
                                "image_id": image_id,
                                "class": obj_class,
                                "edit_type": edit_type,
                                "from_attribute": from_val,
                                "to_attribute": to_val,
                                "input_image_path": "./"
                                + self.dataset_path
                                + "/"
                                + obj_class
                                + "/"
                                + "0" * (12 - len(str(image_id)))
                                + str(image_id)
                                + ".jpg",
                            },
                        )

        # Create a pandas DataFrame from the records
        self.edit_dataset = pd.DataFrame(records)
        self.edit_dataset.to_csv(pandas_path, index=False)

    @staticmethod
    def get_edit(edit_id: int, edit_dataset: pd.DataFrame) -> Edit:
        if edit_id in edit_dataset.index:
            edit = edit_dataset.loc[edit_id]
            return Edit(
                edit_id=edit["edit_id"],
                image_id=edit["image_id"],
                image_path=edit["input_image_path"],
                category=edit["class"],
                edit_type=EditType(edit["edit_type"]),
                from_attribute=edit["from_attribute"],
                to_attribute=edit["to_attribute"],
            )

        error_msg = f"No edit found with edit_id: {edit_id}"
        raise ValueError(error_msg)

    def get_all_edits_image_id(self, image_id: str) -> pd.DataFrame:
        return self.edit_dataset[self.edit_dataset["image_id"] == image_id]

    def get_all_edits_ms_coco_class(self, ms_coco_class: str) -> pd.DataFrame:
        return self.edit_dataset[self.edit_dataset["class"] == ms_coco_class]

    def execute_pipeline(
        self,
        models: list[PromptableImageEditingModel],
    ) -> None:
        for model in models:
            logging.info("Running model: %s", model.get_model_name())
            for idx in self.edit_dataset.index:
                edit = self.get_edit(idx, self.edit_dataset)
                prompt = self.generate_prompt(edit)
                logging.info("prompt: %s", prompt)
                logging.info("image_path: %s", edit.image_path)
                model.edit(prompt, edit.image_path)

    @staticmethod
    def generate_prompt(edit: Edit) -> str:
        prompt_formats = {
            "object_addition": "Add a {to} to the image",
            "positional_addition": "Add a {to} the {category}",
            "size": "Change the size of {category} to {to}",
            "shape": "Change the shape of {category} to {to}",
            "alter_parts": "{to} to {category}",
            "color": "Change the color of {category} to {to}",
            "object_removal": "Remove {category}",
            "object_replacement": "Replace {from_} with {to}",
            "position_replacement": "Move {from_} to {to}",
            "object_duplication": "Duplicate {category}",
            "texture": "Change the texture of {category} to {to}",
            "action": "{category} doing {to}",
            "viewpoint": "Change the viewpoint to {to}",
            "background": "Change the background to {to}",
            "style": "Change the style of {category} to {to}",
        }

        prompt_format = prompt_formats.get(edit.edit_type)
        if prompt_format:
            return prompt_format.format(
                category=edit.category,
                to=edit.to_attribute,
                from_=edit.from_attribute
                if hasattr(edit, "from_attribute")
                else "",
            )
        error_msg = f"Edit type {edit.edit_type} is not implemented"
        raise ValueError(error_msg)
