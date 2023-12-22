import json
from pathlib import Path

import pandas as pd

from pixlens.evaluation import interfaces
from pixlens.editing.interfaces import PromptableImageEditingModel


# create a class that will parse a json object to get some edit instructions
# as a init function it receives the json object file path
# in the init it calls a private function that parses the json object
class EvaluationPipeline:
    def __init__(self, json_object_path: str, dataset_path: str) -> None:
        self.json_object_path = json_object_path
        self.edit_dataset: pd.DataFrame
        self.dataset_path = dataset_path
        self._parse_json_object()

    def _parse_json_object(self) -> None:
        with open(self.json_object_path) as json_file:
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
                    for from_val, to_val in zip(from_values, to_values):
                        records.append(
                            {
                                "edit_id": len(records),
                                "image_id": image_id,
                                "class": obj_class,
                                "edit_type": edit_type,
                                "from": from_val,
                                "to": to_val,
                                "edit_path": self.dataset_path + "/" + image_id,
                            }
                        )

        # Create a pandas DataFrame from the records
        self.edit_dataset = pd.DataFrame(records)

    def __print__(self) -> None:
        print(self.edit_dataset.head())

    # getter edit from edit id
    def get_edit_from_edit_id(self, edit_id: int) -> interfaces.Edit:
        edit = self.edit_dataset.loc[edit_id]
        return interfaces.Edit(
            edit_id=edit["edit_id"],
            image_id=edit["image_id"],
            image_path=str(
                Path(
                    self.dataset_path,
                    edit["class"],
                    f"000000{str(edit['image_id'])}.png",
                )
            ),
            category=edit["class"],
            edit_type=interfaces.EditType(edit["edit_type"]),
            _from=edit["from"],
            _to=edit["to"],
        )

    def execute_pipeline(
        self, models: list[PromptableImageEditingModel]
    ) -> None:
        for model in models:
            for idx in self.edit_dataset.index:
                edit = self.get_edit_from_edit_id(idx)
                edit_type = edit.edit_type
                prompt = ""
                output = model.edit(prompt, edit.image_path)
                raise NotImplementedError


path_to_json = "pixlens//editval//object.json"
path_to_dataset = "editval_instances"
eval = EvaluationPipeline(path_to_json, path_to_dataset)
breakpoint()
