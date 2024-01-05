import argparse
from pathlib import Path

from pixlens.editing import load_editing_model_from_yaml
from pixlens.editing.interfaces import ImageEditingPromptType
from pixlens.utils import utils

parser = argparse.ArgumentParser(
    description="PixLens - Evaluate & understand image editing models",
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
    help=("Path to YAML containing the model configuration"),
    required=True,
)


def main() -> None:
    args = parser.parse_args()

    model = load_editing_model_from_yaml(args.model_params_yaml)

    # check if args.in defined
    in_path = "example_input.jpg"
    edit_info = None
    if args.input is None:
        url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
        image = utils.download_image(url)
        image.save(in_path)

        type_to_prompt = {
            ImageEditingPromptType.INSTRUCTION: "turn him into a cyborg",
            ImageEditingPromptType.DESCRIPTION: (
                "A photo of the sculpture of David[SEP]A photo of a cyborg"
            ),
        }
        prompt = type_to_prompt[model.prompt_type]

    else:
        in_path = args.input
        prompt = args.prompt

    output = model.edit(prompt, in_path)
    if args.input is None:
        Path(in_path).unlink()

    output.save(args.output)


if __name__ == "__main__":
    main()
