import argparse
from pathlib import Path

from pixlens.editing import controlnet, pix2pix, diffedit
from pixlens.utils import utils
from pixlens.evaluation.interfaces import Edit, EditType

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models",
)
parser.add_argument(
    "--model",
    type=str,
    default="InstructPix2Pix",
    help=("Image editing model: pix2pix, dreambooth, etc."),
)
parser.add_argument(
    "--output",
    required=True,
    type=str,
    help=("Path of output image"),
)
parser.add_argument(
    "--input",
    type=str,
    help="Path of input image",
    nargs="?",
    default=None,
)

parser.add_argument("--prompt", type=str, help=("Prompt with edit instruction"))
parser.add_argument(
    "--model_params_yaml",
    type=str,
    help=("Path to YAML containing model params"),
    required=True,
)

NAME_TO_MODEL: dict[
    str,
    type[controlnet.ControlNet]
    | type[pix2pix.Pix2pix]
    | type[diffedit.DiffEdit],
] = {
    "ControlNet": controlnet.ControlNet,
    "InstructPix2Pix": pix2pix.Pix2pix,
    "DiffEdit": diffedit.DiffEdit,
}

MODEL_IS_INSTRUCTIVE: dict[
    str,
    bool,
] = {
    "ControlNet": True,
    "InstructPix2Pix": True,
    "DiffEdit": False,
}


def main() -> None:
    args = parser.parse_args()

    # check if args.in defined
    in_path = "example_input.jpg"
    if args.input is None:
        url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
        image = utils.download_image(url)
        image.save(in_path)

        if MODEL_IS_INSTRUCTIVE[args.model]:
            prompt = "turn him into a cyborg"
        else:
            prompt = "A photo of the sculpture of David[SEP]A photo of a cyborg"
    else:
        in_path = args.input
        prompt = args.prompt

    model = NAME_TO_MODEL[args.model].from_yaml(args.model_params_yaml)

    output = model.edit(prompt, in_path)
    if args.input is None:
        Path(in_path).unlink()

    output.save(args.output)


if __name__ == "__main__":
    main()
