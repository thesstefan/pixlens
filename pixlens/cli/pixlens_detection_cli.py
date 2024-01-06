import argparse

import numpy as np
from PIL import Image

from pixlens.detection import load_detect_segment_model_from_yaml
from pixlens.visualization import annotation

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models",
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
    help=("Path to YAML containing the model configuration"),
    required=True,
)


def main() -> None:
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    model = load_detect_segment_model_from_yaml(args.model_params_yaml)

    segmentation_output, detection_output = model.detect_and_segment(
        args.prompt,
        image,
    )

    image_source = np.asarray(image)

    annotated_image = annotation.annotate_detection_output(
        image_source,
        detection_output,
    )

    masked_annotated_image = annotation.annotate_mask(
        segmentation_output.masks,
        annotated_image,
    )
    masked_annotated_image.save(args.out_image)


if __name__ == "__main__":
    main()
