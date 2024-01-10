import argparse

import torch

from pixlens.editing import (
    controlnet,
    instruct_pix2pix,
    load_editing_model_from_yaml,
)
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation.operations.disentanglement_operation.disentanglement import (
    Disentanglement,
)

parser = argparse.ArgumentParser(description="Evaluate PixLens Editing Model")
parser.add_argument(
    "--model_params_yaml",
    type=str,
    help=("Path to YAML containing the model configuration"),
    required=True,
)


def main() -> None:
    args = parser.parse_args()
    model: PromptableImageEditingModel
    disentangle = Disentanglement(
        json_file_path="disentanglement_json/objects_textures_sizes_colors_styles_test.json",
        image_data_path="editval_instances",
    )
    model = load_editing_model_from_yaml(args.model_params_yaml)
    disentangle.evaluate_model(model=model)


if __name__ == "__main__":
    main()
