import itertools
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from pixlens.editing import interfaces as editing_interfaces
from pixlens.evaluation.operations.disentanglement_operation import utils
from pixlens.evaluation.operations.disentanglement_operation.disentangle_model import (  # noqa: E501
    Classifier,
)
from pixlens.utils.utils import get_cache_dir


class Disentanglement:
    def __init__(self, json_file_path: str, image_data_path: str) -> None:
        """Initialize of datasets and variables.

        Obj_dataset represents the tensors of the form object with
        attribute_type attribute. Att dataset represents the tensors of what
        the model thinks the attribute looks like.
        Dataset is the final dataset that essentially combines the two datasets.
        """
        self.dataset = pd.DataFrame(
            columns=[
                "attribute_type",
                "object",
                "attribute_old",
                "attribute_new",
                "z_start",
                "z_positive_attribute",
                "z_negative_attribute",
                "z_end",
                "norm",
            ],
        )
        self.obj_dataset: pd.DataFrame = pd.DataFrame(
            columns=["attribute_type", "attribute", "object", "z"],
        )
        self.att_dataset: pd.DataFrame = pd.DataFrame(
            columns=["attribute_type", "attribute", "z"],
        )
        self.model: editing_interfaces.PromptableImageEditingModel
        self.json_file_path: Path = Path(json_file_path)
        self.image_data_path: Path = Path(image_data_path)
        self.data_attributes, self.objects = self.load_attributes_and_objects()
        self.generate_images = False

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

    def evaluate_model(
        self,
        model: editing_interfaces.PromptableImageEditingModel,
        *,
        generate_images: bool = False,
    ) -> None:
        """Evaluate the model.

        After initializing the model, checks what was the last thing computed
        and starts the process from there. If nothing is computed, first
        generates the dataset. Then it computes the intra sample evaluation,
        the inter sample and intra attribute evaluation and finally
        the inter attribute
        Args:
            model: The model to evaluate.
            generate_images: Whether to generate images or not.
        """
        self.init_model(model)
        self.generate_images = generate_images
        self.results_path = (
            get_cache_dir()
            / Path("models--" + self.model.get_model_name())
            / Path("disentanglement/")
            / Path("results.json")
        )
        self.results = {}
        if self.results_path.exists():
            with Path.open(self.results_path) as file:
                self.results = json.load(file)
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
        if "Inter_attribute" not in self.results[self.model.get_model_name()]:
            logging.info("Doing inter attribute evaluation")
            self.results[self.model.get_model_name()][
                "Inter_attribute"
            ] = self.inter_attribute()
            with Path.open(self.results_path, "w") as file:
                json.dump(self.results, file, indent=4)

    def check_if_pd_dataset_existent(self) -> bool:
        cache_dir = get_cache_dir()
        parent_folder = (
            cache_dir
            / Path("models--" + self.model.get_model_name())
            / Path("disentanglement/")
        )
        parent_folder.mkdir(parents=True, exist_ok=True)
        self.final_dataset_path = parent_folder / Path("disentanglement.pkl")
        return self.final_dataset_path.exists()

    def generate_dataset(self) -> None:
        """Generate the dataset.

        If the dataset is not existent, first calls the generate all latents
        for image to create the obj_dataset and att_dataset.
        Then calls generate_final_dataset which combines those two datasets
        into the final dataset.
        """
        boolean = self.check_if_pd_dataset_existent()
        if boolean:
            logging.info("Loading existing dataset")
            self.dataset = pd.read_pickle(self.final_dataset_path)  # noqa: S301
        else:
            logging.info("Generating dataset")
            # For now we only use a white image
            self.obj_dataset_path = (
                self.final_dataset_path.parent / "obj_dataset.pkl"
            )
            self.att_dataset_path = (
                self.final_dataset_path.parent / "att_dataset.pkl"
            )
            if not self.obj_dataset_path.exists():
                image_file = self.image_data_path / Path(
                    "000000000002.jpg",
                )
                self.generate_all_latents_for_image(image_file)
            else:
                with Path.open(self.obj_dataset_path, "rb") as file:
                    self.object_dataset = joblib.load(file)
                with Path.open(self.att_dataset_path, "rb") as file:
                    self.att_dataset = joblib.load(file)

            self.generate_final_dataset()
        self.dataset["z_end"] = self.dataset["z_end"].apply(
            lambda x: torch.tensor(x),  # type: ignore[arg-type, return-value]
        )
        self.dataset["z_start"] = self.dataset["z_start"].apply(
            lambda x: torch.tensor(x),  # type: ignore[arg-type, return-value]
        )
        self.dataset["z_positive_attribute"] = self.dataset[
            "z_positive_attribute"
        ].apply(
            lambda x: torch.tensor(x),  # type: ignore[arg-type, return-value]
        )

        self.dataset["z_negative_attribute"] = self.dataset[
            "z_negative_attribute"
        ].apply(
            lambda x: torch.tensor(x),  # type: ignore[arg-type, return-value]
        )

    def generate_all_latents_for_image(self, image_path: Path) -> None:
        """Generate obj_dataset and att_dataset for a given image."""
        data_to_append = []
        str_img_path = str(image_path)
        for attribute in list(self.data_attributes.keys()):
            att_data_to_append = self.generate_reference_attribute_latents(
                attribute,
                str_img_path,
            )
            self.att_dataset = pd.concat(
                [self.att_dataset, pd.DataFrame(att_data_to_append)],
                ignore_index=True,
            )
            self.att_dataset.to_pickle(self.att_dataset_path)
            for o, a in list(
                itertools.product(
                    self.objects,
                    self.data_attributes[attribute],
                ),
            ):
                prompt = utils.get_prompt(o, a)
                z = self.model.get_latent(
                    prompt=prompt,
                    image_path=str_img_path,
                )
                if self.generate_images:
                    image = self.model.edit_image(
                        prompt=prompt,
                        image_path=str_img_path,
                    )
                    image.save(
                        get_cache_dir()
                        / Path("models--" + self.model.get_model_name())
                        / Path("000000000000")
                        / Path(prompt + ".png"),
                    )
                data_to_append.append(
                    {
                        "attribute_type": attribute,
                        "attribute": a,
                        "object": o,
                        "z": z.flatten(),
                    },
                )
        self.obj_dataset = pd.concat(
            [self.obj_dataset, pd.DataFrame(data_to_append)],
            ignore_index=True,
        )
        self.att_dataset.to_pickle(self.att_dataset_path)
        self.obj_dataset.to_pickle(self.obj_dataset_path)

    def generate_reference_attribute_latents(
        self,
        attribute: str,
        image_path: str,
    ) -> list:
        """Generate the reference attribute latents for a given image.

        This is essentially the latents for the prompts "green", "steel", etc.
        """
        data = []
        for a in self.data_attributes[attribute]:
            z = self.model.get_latent(prompt=a, image_path=image_path)
            if self.generate_images:
                image = self.model.edit_image(prompt=a, image_path=image_path)
                image.save(
                    get_cache_dir()
                    / Path("models--" + self.model.get_model_name())
                    / Path("000000000000")
                    / Path(a + ".png"),
                )
            data.append(
                {
                    "attribute_type": attribute,
                    "attribute": a,
                    "z": z.flatten(),
                },
            )
        return data

    def generate_final_dataset(self) -> None:
        data_to_append = []
        for attribute in list(self.data_attributes.keys()):
            for a1, a2 in itertools.permutations(
                self.data_attributes[attribute],
                2,
            ):
                for o in self.objects:
                    z_start_tensor = self.obj_dataset[
                        (self.obj_dataset["object"] == o)
                        & (self.obj_dataset["attribute"] == a1)
                    ]["z"].to_numpy()[0]  # This is a tensor

                    z_end_tensor = self.obj_dataset[
                        (self.obj_dataset["object"] == o)
                        & (self.obj_dataset["attribute"] == a2)
                    ]["z"].to_numpy()[0]  # This is a tensor

                    z_positive_attribute_tensor = self.att_dataset[
                        self.att_dataset["attribute"] == a2
                    ]["z"].to_numpy()[0]  # This is a tensor

                    z_negative_attribute_tensor = self.att_dataset[
                        self.att_dataset["attribute"] == a1
                    ]["z"].to_numpy()[0]  # This is a tensor
                    z_start = (
                        z_start_tensor.cpu().numpy()
                        if z_start_tensor.is_cuda
                        else z_start_tensor.numpy()
                    )
                    z_end = (
                        z_end_tensor.cpu().numpy()
                        if z_end_tensor.is_cuda
                        else z_end_tensor.numpy()
                    )
                    z_positive_attribute = (
                        z_positive_attribute_tensor.cpu().numpy()
                        if z_positive_attribute_tensor.is_cuda
                        else z_positive_attribute_tensor.numpy()
                    )
                    z_negative_attribute = (
                        z_negative_attribute_tensor.cpu().numpy()
                        if z_negative_attribute_tensor.is_cuda
                        else z_negative_attribute_tensor.numpy()
                    )

                    data_to_append.append(
                        {
                            "attribute_type": attribute,
                            "object": o,
                            "attribute_old": a1,
                            "attribute_new": a2,
                            "z_start": z_start,
                            "z_positive_attribute": z_positive_attribute,
                            "z_negative_attribute": z_negative_attribute,
                            "z_end": z_end,
                            "norm": None,
                        },
                    )
        self.dataset = pd.concat(
            [self.dataset, pd.DataFrame(data_to_append)],
            ignore_index=True,
        )
        self.dataset.to_pickle(self.final_dataset_path)

    def intra_sample_evaluation(self) -> tuple[dict[str, float], float]:
        """Compute the intra sample evaluation.

        For every row of self.dataset computes
        || z_end - (z_start + z_positive_attribute - z_negative_attribute) ||.
        """
        norms_per_attribute_type, all_norms = utils.compute_norms(
            self.dataset,
            self.data_attributes,
        )
        avg_norms_per_attribute_type: dict[str, float] = {
            attr_type: float(np.mean(norms))
            for attr_type, norms in norms_per_attribute_type.items()
        }

        overall_avg_norm = np.mean(all_norms)

        return avg_norms_per_attribute_type, overall_avg_norm

    def inter_sample_and_intra_attribute(self) -> dict:
        """Compute the inter sample and intra attribute evaluation.

        For every pair of attributes (of the same attribute type) computes the
        direction vectors of the different objects with those edits and
        then computes the average cosine similarity and average angle between
        those direction vectors.
        """
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
                    ) = utils.compute_attribute_directions(self.dataset, a1, a2)
                    angles.append(avg_angle)
                    cos_similarities.append(avg_cos_similarity)
                    results[attribute_type][a1 + "_" + a2] = {
                        "Average Cosine Similarity": avg_cos_similarity,
                        "Average Angle": avg_angle,
                    }
            results[attribute_type]["Average Cosine Similarity"] = np.mean(
                cos_similarities,
            )
            results[attribute_type]["Average Angle"] = np.mean(angles)
        return results

    def inter_attribute(self) -> float:
        """Compute the inter attribute evaluation."""
        classifier = Classifier(self.dataset, self.final_dataset_path.parent)
        classifier.prepare_data()
        classifier.train_classifier(num_epochs=50)
        classifier.save_checkpoint(classifier.checkpoint_path)
        acc = classifier.evaluate_classifier()
        classifier.plot_confusion_matrix()
        return acc
