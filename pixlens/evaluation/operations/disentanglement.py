import json
import logging
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cosine

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
                "norm",
            ],
        )
        self.model: editing_interfaces.PromptableImageEditingModel
        self.json_file_path: Path = Path(json_file_path)
        self.image_data_path: Path = Path(image_data_path)
        self.data_attributes, self.objects = self.load_attributes_and_objects()
        self.results_path = Path("results_disentanglement.json")
        self.results = {}
        if self.results_path.exists():
            self.results = json.load(self.results_path.open())

    def evaluate_model(
        self,
        model: editing_interfaces.PromptableImageEditingModel,
    ) -> None:
        self.init_model(model)
        if self.model.get_model_name() not in self.results:
            self.results[self.model.get_model_name()] = {}
        self.generate_dataset()
        if "Avg_norm" not in self.results[self.model.get_model_name()]:
            logging.info("Doing intra sample evaluation")
            att_norms, avg_norm = self.intra_sample_evaluation()
            self.results[self.model.get_model_name()]["Avg_norm"] = avg_norm
            self.results[self.model.get_model_name()][
                "Avg_norm_per_attribute"
            ] = att_norms

        for attribute in self.data_attributes:
            logging.info(
                "Average norm for attribute %s : %f",
                attribute,
                self.results[self.model.get_model_name()][
                    "Avg_norm_per_attribute"
                ][attribute],
            )
        with Path.open(self.results_path, "w") as file:
            json.dump(self.results, file, indent=4)
        logging.info(
            "Average norm overall: %f",
            self.results[self.model.get_model_name()]["Avg_norm"],
        )
        logging.info("Doing  inter sample and intra attirbute evaluation")
        if (
            "Inter_sample_and_intra_attribute"
            not in self.results[self.model.get_model_name()]
        ):
            self.results[self.model.get_model_name()][
                "Inter_sample_and_intra_attribute"
            ] = self.inter_sample_and_intra_attribute()
            with Path.open(self.results_path, "w") as file:
                json.dump(self.results, file, indent=4)

    def check_if_pd_dataset_existent(self) -> tuple[str, bool]:
        cache_dir = get_cache_dir()
        pandas_path = (
            cache_dir
            / Path("models--" + self.model.get_model_name())
            / "disentanglement.pkl"
        )
        return str(pandas_path), pandas_path.exists()

    def generate_dataset(self) -> None:
        pandas_path, boolean = self.check_if_pd_dataset_existent()
        if boolean:
            logging.info("Loading existing dataset")
            self.dataset = pd.read_pickle(pandas_path)
        else:
            logging.info("Generating dataset")
            # For now we only use the first image
            image_file = self.image_data_path / Path(
                "000000000000.jpg",
            )  # / Path("dog/000000001025.jpg")
            self.generate_all_latents_for_image(image_file)
            self.dataset["norm"] = None
            self.dataset.to_pickle(pandas_path)

    def generate_all_latents_for_image(self, image_path: Path) -> None:
        data_to_append = []
        cropped_image = self.crop_image_to_min_dimensions(
            image_path,
        )  # TODO: this should be done somewhere else, the load image -> crop -> save -> delete. It would be more efficient to crop the image inside the model.edit() maybe?
        cropped_path = image_path.parent / "cropped.png"
        cropped_image.save(cropped_path)
        # str_img_path = str(cropped_path)
        str_img_path = str(image_path)
        z_0 = self.model.get_latent(prompt="", image_path=str_img_path)
        for attribute in list(self.data_attributes.keys()):
            for o0, o1, a0, a1 in self.generate_ordered_unique_combinations(
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
        object_pairs = set(permutations(objects, 2))
        attribute_pairs = set(permutations(attributes, 2))

        object_pairs = {tuple(sorted(pair)) for pair in object_pairs}
        attribute_pairs = {tuple(sorted(pair)) for pair in attribute_pairs}
        return [
            (*obj_pair, *attr_pair)
            for obj_pair in object_pairs
            for attr_pair in attribute_pairs
        ]

    @staticmethod
    def get_prompt(object_: str, attribute: str, attribute_type: str) -> str:
        # return f"Replace the dog with a {attribute} {object_}"
        return f"Add a {attribute} {object_} to the center of the image"

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

    def compute_norms(self) -> tuple[dict[str, list[float]], list]:
        norms_per_attribute_type: dict[str, list[float]] = {
            attr_type: [] for attr_type in self.data_attributes
        }
        all_norms: list[float] = []
        for _, row in self.dataset.iterrows():
            norm = torch.norm(
                row["z_y"] - (row["z_2"] + row["z_1"] - row["z_neg"]),
            )
            row["norm"] = norm.item()
            norms_per_attribute_type[row["attribute_type"]].append(norm.item())
            all_norms.append(norm.item())

        return norms_per_attribute_type, all_norms

    def intra_sample_evaluation(self) -> tuple[dict[str, float], float]:
        norms_per_attribute_type, all_norms = self.compute_norms()

        avg_norms_per_attribute_type: dict[str, float] = {
            attr_type: float(np.mean(norms))
            for attr_type, norms in norms_per_attribute_type.items()
        }

        overall_avg_norm = np.mean(all_norms)

        return avg_norms_per_attribute_type, overall_avg_norm

    def compute_attribute_directions(
        self,
        a1: str,
        a2: str,
    ) -> tuple[float, float]:
        filtered_rows = self.dataset[
            (self.dataset["attribute_old"] == a1)
            & (self.dataset["attribute_new"] == a2)
        ]

        # Compute vectors
        vectors = []
        for _, row in filtered_rows.iterrows():
            v1 = row["z_y"] - row["z_2"]
            v2 = row["z_1"] - row["z_neg"]
            vectors.extend([v1, v2])

        cos_similarities = []
        angles = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                cos_sim = 1 - cosine(
                    vectors[i].cpu().numpy(),
                    vectors[j].cpu().numpy(),
                )  # cosine similarity
                angle_rad = np.arccos(cos_sim)  # angle in radians

                cos_similarities.append(cos_sim)
                angles.append(angle_rad)

        avg_cos_similarity = np.mean(cos_similarities)
        avg_angle = np.mean(angles)

        return avg_cos_similarity, avg_angle

    def inter_sample_and_intra_attribute(self) -> dict:
        results: dict = {}

        # Iterate over each attribute type
        for attribute_type, attributes in self.data_attributes.items():
            cos_similarities = []
            angles = []
            results[attribute_type] = {}
            for i in range(len(attributes)):
                for j in range(i + 1, len(attributes)):
                    a1, a2 = attributes[i], attributes[j]
                    (
                        avg_cos_similarity,
                        avg_angle,
                    ) = self.compute_attribute_directions(a1, a2)
                    angles.append(avg_angle)
                    cos_similarities.append(avg_cos_similarity)
                    results[attribute_type][(a1, a2)] = {
                        "Average Cosine Similarity": avg_cos_similarity,
                        "Average Angle": avg_angle,
                    }
            results[attribute_type]["Average Cosine Similarity"] = np.mean(
                cos_similarities,
            )
            results[attribute_type]["Average Angle"] = np.mean(angles)
        return results


import torch
from pixlens.editing.pix2pix import Pix2pix
from pixlens.editing.controlnet import ControlNet

disentangle = Disentanglement(
    json_file_path="objects_textures_sizes_colors_styles_test.json",
    image_data_path="editval_instances",
)
disentangle.evaluate_model(Pix2pix(device=torch.device("cuda")))
