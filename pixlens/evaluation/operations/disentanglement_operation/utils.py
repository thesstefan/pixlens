from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import cosine


def compute_attribute_directions(
    dataset: pd.DataFrame,
    a1: str,
    a2: str,
) -> tuple[float, float]:
    filtered_rows = dataset[
        (dataset["attribute_old"] == a1) & (dataset["attribute_new"] == a2)
    ]

    # Compute vectors
    vectors = []
    for _, row in filtered_rows.iterrows():
        vectors.append(row["z_end"] - row["z_start"])
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


def get_prompt(object_: str, attribute: str) -> str:
    return f"{attribute} {object_}"


def crop_image_to_min_dimensions(
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


def compute_norms(
    dataset: pd.DataFrame,
    data_attributes: dict,
) -> tuple[dict[str, list[float]], list]:
    norms_per_attribute_type: dict[str, list[float]] = {
        attr_type: [] for attr_type in data_attributes
    }
    all_norms: list[float] = []
    for _, row in dataset.iterrows():
        norm = torch.norm(
            row["z_end"]
            - (
                row["z_positive_attribute"]
                + row["z_negative_attribute"]
                - row["z_start"]
            ),
        )
        row["norm"] = norm.item()
        norms_per_attribute_type[row["attribute_type"]].append(norm.item())
        all_norms.append(norm.item())
    return norms_per_attribute_type, all_norms
