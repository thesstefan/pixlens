import argparse
import logging

import numpy as np
import torch
from PIL import Image

from pixlens.eval import grounded_sam, owl_vit_sam
from pixlens.visualization import annotation

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models",
)
parser.add_argument(
    "--object-detection",
    type=str,
    default="GroundedSAM",
    help=("Detector, either GroundedSAM or OwlViT+SAM"),
)
parser.add_argument("--out", type=str, help=("Path of output annotated image"))
parser.add_argument("--image", type=str, help=("Image to detect objects in"))
parser.add_argument("--prompt", type=str, help=("Prompt to guide detection"))
parser.add_argument(
    "--detection-threshold",
    type=float,
    default=0.3,
    help=("Detection threshold for object detection"),
)


NAME_TO_MODEL = {
    "GroundedSAM": grounded_sam.GroundedSAM,
    "OwlViT+SAM": owl_vit_sam.OwlViTSAM,
}


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Using device: %s", device)

    model = NAME_TO_MODEL[args.object_detection](
        device=device,
        detection_confidence_threshold=args.detection_threshold,
    )

    segmentation_output, detection_output = model.detect_and_segment(
        args.prompt,
        args.image,
    )

    image_source = np.asarray(Image.open(args.image).convert("RGB"))

    annotated_image = annotation.annotate_detection_output(
        image_source,
        detection_output,
    )

    masked_annotated_image = annotation.annotate_mask(
        segmentation_output.masks,
        annotated_image,
    )
    masked_annotated_image.save(args.out)


if __name__ == "__main__":
    main()
