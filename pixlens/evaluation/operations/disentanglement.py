from typing import Tuple

import pandas as pd
import json
from pathlib import Path
from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.editing import interfaces as editing_interfaces


class Disentanglement:
    def __init__(self, json_file_path: str, image_data_path: str) -> None:
        self.dataset: pd.DataFrame
        self.model: editing_interfaces.PromptableImageEditingModel
        self.json_file_path: Path = Path(json_file_path)
        self.image_data_path: Path = Path(image_data_path)
        self.data_attributes, self.obejcts = self.load_attributes_and_objects()

    def evaluate_model(
        self,
        model: editing_interfaces.PromptableImageEditingModel,
    ) -> None:
        self.init_model(model)

    def generate_dataset(self) -> None:
        for image_class_dir in self.image_data_path.iterdir():
            if image_class_dir.is_dir():
                for image_file in image_class_dir.iterdir():
                    if image_file.is_file():
                        self.generate_all_latents_for_image(image_file)

    def generate_all_latents_for_image(self, image_path) -> None:
        z_0 = self.model.get_latent(prompt="", image_path=image_path)
        for attribute in list(self.data_attributes.keys()):
            for o0, a0, o1, a1 in self.generate_ordered_unique_combinations(
                self.obejcts,
                self.data_attributes[attribute],
            ):
                prompt1 = self.get_prompt(o0, a1, attribute)
                prompt2 = self.get_prompt(o1, a0, attribute)
                prompt3 = self.get_prompt(o0, a0, attribute)
                prompty = self.get_prompt(o1, a1, attribute)
                z_1 = (
                    self.model.get_latent(prompt=prompt1, image_path=image_path)
                    - z_0
                )
                z_2 = (
                    self.model.get_latent(prompt=prompt2, image_path=image_path)
                    - z_0
                )
                z_3 = (
                    self.model.get_latent(prompt=prompt3, image_path=image_path)
                    - z_0
                )
                z_y = (
                    self.model.get_latent(prompt=prompty, image_path=image_path)
                    - z_0
                )
        return None

    def init_model(
        self, model: editing_interfaces.PromptableImageEditingModel
    ) -> None:
        self.model = model

    def load_attributes_and_objects(self) -> dict:
        with open(self.json_file_path, "r") as file:
            data_loaded = json.load(file)
        objects = data_loaded["objects"]
        data_loaded.pop("objects")
        return data_loaded, objects

    @staticmethod
    def generate_ordered_unique_combinations(
        objects, attributes
    ) -> list[Tuple[str, str, str, str]]:
        # Generate all permutations of objects and attributes
        object_perms = permutations(objects, 2)
        attribute_perms = permutations(attributes, 2)

        # Combine each unique object pair with each unique attribute pair in the desired format
        return [
            (obj_pair[0], attr_pair[0], obj_pair[1], attr_pair[1])
            for obj_pair in object_perms
            for attr_pair in attribute_perms
        ]

    @staticmethod
    def get_prompt(object_: str, attribute: str, attribute_type: str) -> str:
        return f"Add object {object_} with {attribute} {attribute_type}"
