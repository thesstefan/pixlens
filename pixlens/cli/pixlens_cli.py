import argparse

import torch

from pixlens.eval import grounded_sam

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models"
)

parser.add_argument("--image", type=str, help=("Image to detect objects in"))
parser.add_argument("--prompt", type=str, help=("Prompt to guide detection"))


def main() -> None:
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = grounded_sam.GroundedSAM(device=device)
    segmentation_output, detection_output = model.detect_and_segment(
        args.prompt, args.image
    )


if __name__ == "__main__":
    main()
