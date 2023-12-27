import argparse
from typing import TYPE_CHECKING

import torch

from pixlens.editing import controlnet, pix2pix
from pixlens.evaluation.operations.disentanglement import Disentanglement

if TYPE_CHECKING:
    from pixlens.editing.interfaces import PromptableImageEditingModel
parser = argparse.ArgumentParser(description="Evaluate PixLens Editing Model")
parser.add_argument(
    "--editing-model",
    type=str,
    required=True,
    help="Name of the editing model to use",
)


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda")
    model: PromptableImageEditingModel
    disentangle = Disentanglement(
        json_file_path="objects_textures_sizes_colors_styles_test.json",
        image_data_path="editval_instances",
    )
    if args.editing_model == "InstructPix2Pix":
        model = pix2pix.Pix2pix(pix2pix.Pix2pixType.BASE, device)
    elif args.editing_model == "controlnet":
        model = controlnet.ControlNet(controlnet.ControlNetType.BASE, device)
    else:
        raise NotImplementedError
    disentangle.evaluate_model(model=model)


if __name__ == "__main__":
    main()
