import json
from itertools import permutations
from pathlib import Path

import pandas as pd

from pixlens.editing import interfaces as editing_interfaces
from pixlens.utils.utils import get_cache_dir


class Disentanglement:
    def __init__(self, json_file_path: str, image_data_path: str) -> None:
        self.dataset: pd.DataFrame = pd.DataFrame(
            columns=["z_0", "z_1", "z_2", "z_neg", "z_y"],
        )
        self.model: editing_interfaces.PromptableImageEditingModel
        self.json_file_path: Path = Path(json_file_path)
        self.image_data_path: Path = Path(image_data_path)
        self.data_attributes, self.obejcts = self.load_attributes_and_objects()

    def evaluate_model(
        self,
        model: editing_interfaces.PromptableImageEditingModel,
    ) -> None:
        self.init_model(model)

    def check_if_pd_dataset_existent(self) -> tuple[str, bool]:
        cache_dir = get_cache_dir()
        pandas_path = (
            cache_dir
            / Path(self.model.get_model_name())
            / "disentanglement.csv"
        )
        return str(pandas_path), pandas_path.exists()

    def generate_dataset(self) -> None:
        pandas_path, boolean = self.check_if_pd_dataset_existent()
        if boolean:
            self.dataset = pd.read_csv(pandas_path)
        for image_class_dir in self.image_data_path.iterdir():
            if image_class_dir.is_dir():
                for image_file in image_class_dir.iterdir():
                    if image_file.is_file():
                        self.generate_all_latents_for_image(image_file)
        self.dataset.to_csv(pandas_path)

    def generate_all_latents_for_image(self, image_path: Path) -> None:
        data_to_append = []
        z_0 = self.model.get_latent(prompt="", image_path=str(image_path))
        for attribute in list(self.data_attributes.keys()):
            for o0, a0, o1, a1 in self.generate_ordered_unique_combinations(
                self.obejcts,
                self.data_attributes[attribute],
            ):
                prompt1 = self.get_prompt(o0, a1, attribute)
                prompt2 = self.get_prompt(o1, a0, attribute)
                promptneg = self.get_prompt(o0, a0, attribute)
                prompty = self.get_prompt(o1, a1, attribute)
                z_1 = (
                    self.model.get_latent(prompt=prompt1, image_path=image_path)
                    - z_0
                )
                z_2 = (
                    self.model.get_latent(prompt=prompt2, image_path=image_path)
                    - z_0
                )
                z_neg = (
                    self.model.get_latent(
                        prompt=promptneg,
                        image_path=image_path,
                    )
                    - z_0
                )
                z_y = (
                    self.model.get_latent(prompt=prompty, image_path=image_path)
                    - z_0
                )

                data_to_append.append(
                    {
                        "z_0": z_0.flatten(),
                        "z_1": z_1.flatten(),
                        "z_2": z_2.flatten(),
                        "z_neg": z_neg.flatten(),
                        "z_y": z_y.flatten(),
                    },
                )
        self.dataset = pd.concat(
            [self.dataset, pd.DataFrame(data_to_append)],
            ignore_index=True,
        )

    def init_model(
        self,
        model: editing_interfaces.PromptableImageEditingModel,
    ) -> None:
        self.model = model

    def load_attributes_and_objects(self) -> tuple[dict, list]:
        with Path.open(self.json_file_path) as file:
            data_loaded = json.load(file)
        objects = data_loaded["objects"]
        data_loaded.pop("objects")
        return data_loaded, objects

    @staticmethod
    def generate_ordered_unique_combinations(
        objects: list,
        attributes: list,
    ) -> list[tuple[str, str, str, str]]:
        # Generate all permutations of objects and attributes
        object_perms = permutations(objects, 2)
        attribute_perms = permutations(attributes, 2)
        return [
            (obj_pair[0], attr_pair[0], obj_pair[1], attr_pair[1])
            for obj_pair in object_perms
            for attr_pair in attribute_perms
        ]

    @staticmethod
    def get_prompt(object_: str, attribute: str, attribute_type: str) -> str:
        return f"Add object {object_} with {attribute} {attribute_type}"
