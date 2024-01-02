import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pixlens.editing import interfaces as editing_interfaces
from pixlens.utils.utils import get_cache_dir


class Disentanglement:
    def __init__(self, json_file_path: str, image_data_path: str) -> None:
        self.dataset: pd.DataFrame = pd.DataFrame(
            columns=[
                "attribute_type",
                "attribute_old",
                "attribute_new",
                "z_0",
                "z_1",
                "z_2",
                "z_neg",
                "z_y",
            ],
        )
        self.model: editing_interfaces.PromptableImageEditingModel
        self.json_file_path: Path = Path(json_file_path)
        self.image_data_path: Path = Path(image_data_path)
        self.data_attributes, self.objects = self.load_attributes_and_objects()

    def evaluate_model(
        self,
        model: editing_interfaces.PromptableImageEditingModel,
    ) -> None:
        self.init_model(model)
        pd_data_path, path_exists = self.check_if_pd_dataset_existent()
        if path_exists:
            self.dataset = pd.read_csv(pd_data_path)
        else:
            self.generate_dataset()
        self.intra_sample_evaluation()

    def check_if_pd_dataset_existent(self) -> tuple[str, bool]:
        cache_dir = get_cache_dir()
        pandas_path = (
            cache_dir
            / Path("models--" + self.model.get_model_name())
            / "disentanglement.csv"
        )
        return str(pandas_path), pandas_path.exists()

    def generate_dataset(self) -> None:
        pandas_path, boolean = self.check_if_pd_dataset_existent()
        if boolean:
            self.dataset = pd.read_csv(pandas_path)
        for image_class_dir in self.image_data_path.iterdir():
            if image_class_dir.is_dir():
                image_file = next(
                    image_class_dir.iterdir()
                )  # Let's do only one image for now per class
                if image_file.is_file():
                    self.generate_all_latents_for_image(image_file)
        self.dataset.to_csv(pandas_path)

    def generate_all_latents_for_image(self, image_path: Path) -> None:
        data_to_append = []
        cropped_image = self.crop_image_to_min_dimensions(
            image_path,
        )  # TODO: this should be done somewhere else, the load image -> crop -> save -> delete. It would be more efficient to crop the image inside the model.edit() maybe?
        cropped_path = image_path.parent / "cropped.png"
        cropped_image.save(cropped_path)
        str_img_path = str(cropped_path)
        z_0 = self.model.get_latent(prompt="", image_path=str_img_path)
        for attribute in list(self.data_attributes.keys()):
            for o0, a0, o1, a1 in self.generate_ordered_unique_combinations(
                self.objects,
                self.data_attributes[attribute],
            ):
                prompt1 = self.get_prompt(o0, a1, attribute)
                prompt2 = self.get_prompt(o1, a0, attribute)
                promptneg = self.get_prompt(o0, a0, attribute)
                prompty = self.get_prompt(o1, a1, attribute)
                z_1 = (
                    self.model.get_latent(
                        prompt=prompt1,
                        image_path=str_img_path,
                    )
                    # - z_0
                )
                z_2 = (
                    self.model.get_latent(
                        prompt=prompt2,
                        image_path=str_img_path,
                    )
                    # - z_0
                )
                z_neg = (
                    self.model.get_latent(
                        prompt=promptneg,
                        image_path=str_img_path,
                    )
                    # - z_0
                )
                z_y = (
                    self.model.get_latent(
                        prompt=prompty,
                        image_path=str_img_path,
                    )
                    # - z_0
                )
                self.model.edit(prompt=prompt1, image_path=str_img_path)
                self.model.edit(prompt=prompt2, image_path=str_img_path)
                self.model.edit(prompt=promptneg, image_path=str_img_path)
                self.model.edit(prompt=prompty, image_path=str_img_path)
                data_to_append.append(
                    {
                        "attribute_type": attribute,
                        "attribute_old": a0,
                        "attribute_new": a1,
                        "z_0": z_0.flatten(),
                        "z_1": z_1.flatten(),
                        "z_2": z_2.flatten(),
                        "z_neg": z_neg.flatten(),
                        "z_y": z_y.flatten(),
                    },
                )
        cropped_path.unlink()
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
        return f"Add {object_} with {attribute_type} {attribute}"

    def crop_image_to_min_dimensions(
        self,
        image_path: Path,
        min_width: int = 320,  # min of widths from editval
        min_height: int = 186,  # min of heights from editval
    ) -> Image.Image:
        with Image.open(image_path) as img:
            width, height = img.size
            left = int((width - min_width) / 2)
            top = int((height - min_height) / 2)
            right = int((width + min_width) / 2)
            bottom = int((height + min_height) / 2)

            # Crop the center of the image
            return img.crop((left, top, right, bottom))

    def intra_sample_evaluation(self) -> None:
        raise NotImplementedError

    def compute_norms(self) -> tuple[dict[str, list[float]], list]:
        # Initialize a dictionary to store the norms for each attribute type
        norms_per_attribute_type: dict[str, list[float]] = {
            attr_type: [] for attr_type in self.data_attributes
        }

        # List to store norms for all samples
        all_norms: list[float] = []

        # Iterate over the dataset
        for _, row in self.dataset.iterrows():
            # Compute the norm for the current sample using PyTorch
            norm = torch.norm(
                row["z_y"] - (row["z_2"] + row["z_1"] - row["z_neg"]),
            )

            # Append the norm to the respective attribute type list and to the overall list
            norms_per_attribute_type[row["attribute_type"]].append(norm.item())
            all_norms.append(norm.item())

        return norms_per_attribute_type, all_norms

    def calculate_statistics(self) -> tuple[dict[str, float], float]:
        norms_per_attribute_type, all_norms = self.compute_norms()

        # Compute average norms per attribute type
        avg_norms_per_attribute_type: dict[str, float] = {
            attr_type: np.mean(norms)
            for attr_type, norms in norms_per_attribute_type.items()
        }

        # Compute overall average norm
        overall_avg_norm = np.mean(all_norms)

        return avg_norms_per_attribute_type, overall_avg_norm


import torch
from pixlens.editing.pix2pix import Pix2pix
from pixlens.editing.controlnet import ControlNet

disentangle = Disentanglement(
    json_file_path="objects_textures_sizes_colors_styles_test.json",
    image_data_path="editval_instances",
)
disentangle.evaluate_model(Pix2pix(device=torch.device("cuda")))
