import argparse
import torch
import PIL
import requests

from pixlens.image_editing_models import pix2pix

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models"
)
parser.add_argument(
    "--edit-model",
    type=str,
    default="pix2pix",
    help=("Image editing model: pix2pix, dreambooth, etc."),
)
parser.add_argument("--output", type=str, help=("Path of output image"))
parser.add_argument(
    "--input", type=str, help="Path of input image", nargs="?", default=None
)

parser.add_argument("--prompt", type=str, help=("Prompt with edit instruction"))


def main() -> None:
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # check if args.in defined
    in_path = "example_input.jpg"
    if args.input is None:
        url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"

        def download_image(url):
            image = PIL.Image.open(requests.get(url, stream=True).raw)
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image

        image = download_image(url)
        image.save(in_path)
        prompt = "turn him into cyborg"
    else:
        in_path = args.input
        prompt = args.prompt

    # code to instantiate and run pix2pix
    if args.edit_model == "pix2pix":
        model = pix2pix.Pix2pix(pix2pix.Pix2pixType.BASE, device)
    else:
        raise NotImplementedError

    output = model.edit_image(prompt, in_path)
    output.image.save(args.output)


if __name__ == "__main__":
    main()
