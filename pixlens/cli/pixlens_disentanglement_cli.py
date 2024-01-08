import argparse

import torch

from pixlens.editing import controlnet, instruct_pix2pix
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation.operations.disentanglement_operation.disentanglement import (
    Disentanglement,
)

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
        json_file_path="disentanglement_json/objects_textures_sizes_colors_styles_test.json",
        image_data_path="editval_instances",
    )
    if args.editing_model == "InstructPix2Pix":
        model = instruct_pix2pix.InstructPix2Pix(
            instruct_pix2pix.InstructPix2PixType.BASE,
            device,
        )
    elif args.editing_model == "controlnet":
        model = controlnet.ControlNet(controlnet.ControlNetType.BASE, device)
    else:
        raise NotImplementedError
    disentangle.evaluate_model(model=model)


if __name__ == "__main__":
    main()
