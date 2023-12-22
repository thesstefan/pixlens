import json
from pathlib import Path

import pandas as pd
import torch

from pixlens.evaluation import interfaces
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.editing.pix2pix import Pix2pix


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
    def get_editfrom_attribute_edit_id(self, edit_id: int) -> interfaces.Edit:
        edit = self.edit_dataset.loc[edit_id]
        return interfaces.Edit(
            edit_id=edit["edit_id"],
            image_id=edit["image_id"],
            image_path=str(
                Path(
                    self.dataset_path,
                    edit["class"],
                    f"000000{str(edit['image_id'])}.jpg",
                )
            ),
            category=edit["class"],
            edit_type=interfaces.EditType(edit["edit_type"]),
            from_attribute=edit["from"],
            to_attribute=edit["to"],
        )

    def execute_pipeline(
        self,
        models: list[PromptableImageEditingModel],
    ) -> None:
        for model in models:
            for idx in self.edit_dataset.index:
                edit = self.get_editfrom_attribute_edit_id(idx)
                prompt = self.generate_prompt(edit)
                output = model.edit(prompt, edit.image_path)
                score = 0

    def generate_prompt(self, edit: interfaces.Edit) -> str:
        prompt_formats = {
            "object_addition": "Add a {} to the image",
            "positional_addition": "Add {} the {}",
            "size": "Change the size of {} to {}",
            "shape": "Change the shape of {} to {}",
            "alter_parts": "{} to {}",
            "color": "Change the color of {} to {}",
            "object_change": "Change {} to {}",
            "object_removal": "Remove {}",
            "object_replacement": "Replace {} with {}",
            "position_replacement": "Move {} to {}",
            "object_duplication": "Duplicate {}",
            "texture": "Change the texture of {} to {}",
            "action": "{} doing {}",
            "viewpoint": "Change the viewpoint to {}",
            "background": "Change the background to {}",
            "style": "Change the style of {} to {}",
        }

        prompt_format = prompt_formats.get(edit.edit_type)
        if prompt_format:
            # Handle special cases where from_attribute is needed
            if edit.edit_type in ["object_replacement", "position_replacement"]:
                return prompt_format.format(
                    edit.from_attribute, edit.to_attribute
                )
            else:
                return prompt_format.format(edit.category, edit.to_attribute)
        else:
            raise NotImplementedError(edit.edit_type, "not implemented")


pathto_attribute_json = "pixlens//editval//object.json"
pathto_attribute_dataset = "editval_instances"
eval = EvaluationPipeline(pathto_attribute_json, pathto_attribute_dataset)
model = Pix2pix(device=torch.device("cuda"))
eval.execute_pipeline([model])
