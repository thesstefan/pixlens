import argparse

import torch

from pixlens.detection.grounded_sam import GroundedSAM
from pixlens.detection.owl_vit_sam import OwlViTSAM
from pixlens.editing.controlnet import ControlNet
from pixlens.editing.pix2pix import Pix2pix
from pixlens.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)
from pixlens.evaluation.operations.size import SizeEdit
from pixlens.evaluation.preprocessing_pipeline import PreprocessingPipeline

parser = argparse.ArgumentParser(description="Evaluate PixLens Editing Model")
parser.add_argument(
    "--edit-id",
    default=0,
    type=int,
    required=True,
    help="ID of the edit to be evaluated",
)
parser.add_argument(
    "--editing-model",
    type=str,
    required=True,
    help="Name of the editing model to use",
)
parser.add_argument(
    "--detection-model",
    type=str,
    required=True,
    help="Name of the detection model to use",
)
# Add other necessary arguments if needed


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cpu")

    # Run the Preprocessing Pipeline (if needed)
    preprocessing_pipe = PreprocessingPipeline(
        "./pixlens/editval/object.json",
        "./editval_instances/",
    )
    editing_model = (
        ControlNet(device=device)
        if args.editing_model.lower() == "controlnet"
        else Pix2pix(device=device)
    )
    preprocessing_pipe.execute_pipeline(models=[editing_model])

    # Initialize the EvaluationPipeline
    evaluation_pipeline = EvaluationPipeline(device=device)
    detection_model = (
        GroundedSAM(device=device)
        if args.detection_model.lower() == "groundedsam"
        else OwlViTSAM(device=device)
    )  # Replace with actual model initialization

    evaluation_pipeline.init_editing_model(editing_model)
    evaluation_pipeline.init_detection_model(detection_model)

    # Get the edit object from the dataset using the provided edit ID
    edit = preprocessing_pipe.get_edit(
        args.edit_id,
        evaluation_pipeline.edit_dataset,
    )

    # Get all inputs for the edit
    evaluation_input = evaluation_pipeline.get_all_inputs_for_edit(edit)

    # Do as you please with the evaluation_input
    # For example, you can do:
    print(evaluation_input.prompt)  # noqa: T201
    score = SizeEdit().evaluate_edit(evaluation_input)
    print(score)  # noqa: T201


if __name__ == "__main__":
    main()
