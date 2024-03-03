import argparse

from pixlens.editing import (
    load_editing_model_from_yaml,
)
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation.operations.disentanglement_operation.disentanglement import (
    Disentanglement,
)

parser = argparse.ArgumentParser(description="Evaluate PixLens Editing Model")
parser.add_argument(
    "--model-params-yaml",
    type=str,
    help=("Path to YAML containing the model configuration"),
    required=True,
)


def main() -> None:
    args = parser.parse_args()
    model: PromptableImageEditingModel
    disentangle = Disentanglement(
        json_file_path="disentanglement_files/objects_textures_sizes_colors_styles_extended.json",
        image_data_path="disentanglement_files",
    )
    model = load_editing_model_from_yaml(args.model_params_yaml)
    disentangle.evaluate_model(model=model, generate_images=True)


if __name__ == "__main__":
    main()
