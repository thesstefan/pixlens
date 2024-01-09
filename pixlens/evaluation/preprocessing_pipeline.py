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

            # TODO: add more complex schema validation  # noqa: TD002, FIX002, TD003, E501
            schema = pa.DataFrameSchema(
                {
                    "edit_id": Column(pa.Int),
                    "image_id": Column(pa.Int),
                    "edit_type": Column(pa.String),
                    "class": Column(pa.String),
                    "from_attribute": Column(pa.String, nullable=True),
                    "to_attribute": Column(pa.String, nullable=True),
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

                    # Iterate through "to" values
                    for from_val in from_values:
                        for to_val in to_values:
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
        records = self.add_object_removal(records)
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

    def get_all_edits_edit_type(self, edit_type: str) -> pd.DataFrame:
        return self.edit_dataset[self.edit_dataset["edit_type"] == edit_type]

    def get_edited_image_path(
        self,
        input_image_path: Path,
        prompt: str,
        editing_model: PromptableImageEditingModel,
    ) -> Path:
        return (
            get_cache_dir()
            / editing_model.model_id
            / input_image_path.stem
            / prompt
        ).with_suffix(".png")

    def save_edited_image(
        self,
        edit: Edit,
        prompt: str,
        editing_model: PromptableImageEditingModel,
    ) -> None:
        edited_image_path = self.get_edited_image_path(
            Path(edit.image_path),
            prompt,
            editing_model,
        )

        if edited_image_path.exists():
            logging.info(
                "Image already exists at %s...",
                edited_image_path,
            )

            return

        logging.info("Editing image...")

        edited_image_path.parent.mkdir(parents=True, exist_ok=True)
        edited_image = editing_model.edit_image(prompt, edit.image_path, edit)
        edited_image.save(edited_image_path)

        logging.info("Image saved to %s", edited_image_path)

    def init_model_dir(self, model: PromptableImageEditingModel) -> None:
        model_dir = get_cache_dir() / model.model_id
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        model.to_yaml(model_dir / "model_params.yaml")

    def execute_pipeline(
        self,
        models: list[PromptableImageEditingModel],
    ) -> None:
        for model in models:
            logging.info("Running model: %s", model.get_model_name())

            self.init_model_dir(model)

            for idx in self.edit_dataset.index:
                edit = self.get_edit(idx, self.edit_dataset)
                prompt = model.generate_prompt(edit)

                logging.info("prompt: %s", prompt)
                logging.info("image_path: %s", edit.image_path)

                self.save_edited_image(edit, prompt, model)

                logging.info("")

    def add_object_removal(self, records: list[dict]) -> list[dict]:
        for category_path in Path(self.dataset_path).iterdir():
            if category_path.is_dir():
                for image_path in category_path.iterdir():
                    if image_path.is_file() and image_path.suffix in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        image_id = (
                            image_path.stem.lstrip("0") or "0"
                        )  # Remove leading zeros
                        obj_class = category_path.name
                        edit_type = (
                            "object_removal"  # Assuming edit_type is constant
                        )
                        from_val = (
                            None  # Set appropriate value for from_attribute
                        )
                        to_val = None  # Set appropriate value for to_attribute

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
                                + image_id
                                + ".jpg",
                            },
                        )
        return records
