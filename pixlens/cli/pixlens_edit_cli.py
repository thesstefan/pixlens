import argparse
import logging
from pathlib import Path

import torch

from pixlens.editing import controlnet, pix2pix
from pixlens.utils import utils

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models"
)
parser.add_argument(
    "--edit-model",
    type=str,
    default="pix2pix",
    help=("Image editing model: pix2pix, dreambooth, etc."),
)
parser.add_argument(
    "--output", required=True, type=str, help=("Path of output image")
)
parser.add_argument(
    "--input", type=str, help="Path of input image", nargs="?", default=None
)

parser.add_argument("--prompt", type=str, help=("Prompt with edit instruction"))


def main() -> None:
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # check if args.in defined
    in_path = "example_input.jpg"
    if args.input is None:
        url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
        image = utils.download_image(url)
        image.save(in_path)
        prompt = "turn him into cyborg"
    else:
        in_path = args.input
        prompt = args.prompt

    # code to instantiate and run pix2pix
    if args.edit_model == "pix2pix":
        model = pix2pix.Pix2pix(pix2pix.Pix2pixType.BASE, device)
    elif args.edit_model == "controlnet":
        model = controlnet.ControlNet(controlnet.ControlNetType.BASE, device)
    else:
        raise NotImplementedError

    output = model.edit(prompt, in_path)
    if args.input is None:
        Path(in_path).unlink()

    output.image.save(args.output)


if __name__ == "__main__":
    main()
