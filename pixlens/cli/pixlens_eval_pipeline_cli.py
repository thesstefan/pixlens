import argparse

from pixlens.detection.grounded_sam import GroundedSAM
from pixlens.detection.owl_vit_sam import OwlViTSAM
from pixlens.editing.controlnet import ControlNet
from pixlens.editing.pix2pix import Pix2pix
from pixlens.evaluation.evaluate_models import (
    EvaluationPipeline,
)
from pixlens.evaluation.edit_dataset import PreprocessingPipeline

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

    # Initialize the EvaluationPipeline
    evaluation_pipeline = EvaluationPipeline()
    editing_model = (
        ControlNet()
        if args.editing_model.lower() == "controlnet"
        else Pix2pix()
    )
    detection_model = (
        GroundedSAM
        if args.detection_model.lower() == "groundedsam"
        else OwlViTSAM
    )  # Replace with actual model initialization

    evaluation_pipeline.init_editing_model(editing_model)
    evaluation_pipeline.init_detection_model(detection_model)

    # Get the edit object from the dataset using the provided edit ID
    edit = PreprocessingPipeline.get_edit(
        args.edit_id, evaluation_pipeline.edit_dataset
    )

    # Get all inputs for the edit
    evaluation_input = evaluation_pipeline.get_all_inputs_for_edit(edit)

    breakpoint()
