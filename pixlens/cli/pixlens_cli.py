import argparse

import numpy as np
import torch
from PIL import Image

from pixlens.eval import grounded_sam, owl_vit_SAM
from pixlens.visualization import annotation
parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models"
)
parser.add_argument("--objectdetection", type=str, default="GroundedSAM", help=("Detector, either GroundedSAM or Owl-vitSAM"))
parser.add_argument("--out", type=str, help=("Path of output annotated image"))
parser.add_argument("--image", type=str, help=("Image to detect objects in"))
parser.add_argument("--prompt", type=str, help=("Prompt to guide detection"))


def main() -> None:
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.objectdetection == "GroundedSAM":
        model = grounded_sam.GroundedSAM(device=device)
    elif args.objectdetection == "Owl-vitSAM":
        model = owl_vit_SAM.OwlVitSam(device=device)
    else:
        raise NotImplementedError
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
