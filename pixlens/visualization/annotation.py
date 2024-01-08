from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from groundingdino.util import box_ops, inference
from PIL import Image, ImageDraw

from pixlens.detection import interfaces


# TODO: Decouple this from GroundingDINO, it should work for any bounding boxes.
def annotate_detection_output(
    image_source: npt.NDArray[Any],
    detection_output: interfaces.DetectionOutput,
) -> Image.Image:
    height, width, _ = image_source.shape
    bbox = detection_output.bounding_boxes / torch.Tensor(
        [width, height, width, height],
    )
    bbox_cxcywh = box_ops.box_xyxy_to_cxcywh(bbox)

    annotated_frame = inference.annotate(
        image_source=image_source,
        boxes=bbox_cxcywh,
        logits=detection_output.logits,
        phrases=detection_output.phrases,
    )[..., ::-1]

    return Image.fromarray(annotated_frame)


def annotate_mask(
    masks: torch.Tensor,
    image: Image.Image,
    mask_alpha: float = 0.8,
) -> Image.Image:
    height, width = masks.shape[-2:]

    mask_image = torch.zeros(height, width, 1)
    for mask in masks:
        color = np.concatenate(
            [np.random.random(3), np.array([mask_alpha])],
            axis=0,
        )

        mask_image = torch.max(
            mask_image,
            mask.cpu().reshape(height, width, 1) * color.reshape(1, 1, -1),
        )

    mask_image_pil = Image.fromarray(
        (mask_image.numpy() * 255).astype(np.uint8),
    ).convert("RGBA")

    return Image.alpha_composite(image.convert("RGBA"), mask_image_pil)


def draw_center_of_masses(
    annotated_image: Image.Image,
    ini_center_of_mass: tuple[float, float],
    end_center_of_mass: tuple[float, float],
) -> Image.Image:
    draw = ImageDraw.Draw(annotated_image)
    draw.ellipse(
        [
            ini_center_of_mass[1] - 5,
            ini_center_of_mass[0] - 5,
            ini_center_of_mass[1] + 5,
            ini_center_of_mass[0] + 5,
        ],
        fill="red",
    )
    draw.ellipse(
        [
            end_center_of_mass[1] - 5,
            end_center_of_mass[0] - 5,
            end_center_of_mass[1] + 5,
            end_center_of_mass[0] + 5,
        ],
        fill="blue",
    )

    # draw arrow in direction "ini" -> "end" with the tip at "end"
    draw.line(
        [
            ini_center_of_mass[1],
            ini_center_of_mass[0],
            end_center_of_mass[1],
            end_center_of_mass[0],
        ],
        fill="black",
        width=2,
    )
    return annotated_image
