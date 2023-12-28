import argparse
import logging

import numpy as np
import torch
from PIL import Image

from pixlens.detection import grounded_sam, owl_vit_sam
from pixlens.visualization import annotation

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models"
)
parser.add_argument(
    "--model",
    type=str,
    default="GroundedSAM",
    help=("Detect+Segment model, GroundedSAM or OwlViTSAM"),
)
parser.add_argument(
    "--out_image",
    type=str,
    help=("Path of output annotated image"),
    required=True,
)
parser.add_argument(
    "--image",
    type=str,
    help=("Path of image to detect objects in"),
    required=True,
)
parser.add_argument("--prompt", type=str, help=("Prompt to guide detection"))
parser.add_argument(
    "--model_params_yaml",
    type=str,
    help=("Path to YAML containing model params"),
    required=True,
)

NAME_TO_MODEL: dict[
    str,
    type[grounded_sam.GroundedSAM] | type[owl_vit_sam.OwlViTSAM],
] = {
    "GroundedSAM": grounded_sam.GroundedSAM,
    "OwlViTSAM": owl_vit_sam.OwlViTSAM,
}


def main() -> None:
    args = parser.parse_args()

    model = NAME_TO_MODEL[args.model].from_yaml(args.model_params_yaml)
    segmentation_output, detection_output = model.detect_and_segment(
        args.prompt, args.image
    )

    image_source = np.asarray(Image.open(args.image).convert("RGB"))

    annotated_image = annotation.annotate_detection_output(
        image_source, detection_output
    )

    masked_annotated_image = annotation.annotate_mask(
        segmentation_output.masks, annotated_image
    )
    masked_annotated_image.save(args.out)


if __name__ == "__main__":
    main()
