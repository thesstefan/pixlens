import argparse
import logging

import torch

from pixlens.detection.grounded_sam import GroundedSAM
from pixlens.detection.owl_vit_sam import OwlViTSAM
from pixlens.editing.controlnet import ControlNet
from pixlens.editing.pix2pix import Pix2pix
from pixlens.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)
from pixlens.evaluation.interfaces import EditType
from pixlens.evaluation.operations.color import ColorEdit
from pixlens.evaluation.operations.object_addition import ObjectAddition
from pixlens.evaluation.operations.size import SizeEdit
from pixlens.evaluation.preprocessing_pipeline import PreprocessingPipeline

parser = argparse.ArgumentParser(description="Evaluate PixLens Editing Model")
parser.add_argument(
    "--edit-id",
    type=int,
    required=False,
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
# add edit type argument as a string
parser.add_argument(
    "--edit-type",
    type=str,
    required=False,
    help="Name of the edit type to evaluate",
)


# check arguments function. Check that either edit type or edit id is provided
# and that the edit type is valid by importing the edit type class from
# pixlens/evaluation/interfaces.py and checking that the edit type is in the
# class
#
def check_args(args: argparse.Namespace) -> None:
    if args.edit_id is None and args.edit_type is None:
        error_msg = "Either edit id or edit type must be provided"
        raise ValueError(error_msg) from None
    if args.edit_type is not None:
        try:
            EditType.from_type_name(args.edit_type)
        except ValueError as err:
            error_msg = "Invalid edit type provided"
            raise ValueError(error_msg) from err


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_args(args)

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
    # otherwise use the edit type to get a random edit object from the dataset
    # using the edit type
    if args.edit_id is None:
        all_edits_by_type = preprocessing_pipe.get_all_edits_edit_type(
            args.edit_type,
        )
        # get first edit from the all_edits_by_type dataframe
        random_edit_record = all_edits_by_type.iloc[[4]]
        #     # random_edit_record = all_edits_by_type.sample(n=1)  # noqa: ERA001
        edit = preprocessing_pipe.get_edit(
            (random_edit_record["edit_id"].astype(int).to_numpy()[0]),
            evaluation_pipeline.edit_dataset,
        )
        #
    else:
        edit = preprocessing_pipe.get_edit(
            args.edit_id,
            evaluation_pipeline.edit_dataset,
        )
    logging.info("Running edit: %s", edit.edit_id)
    logging.info("Edit type: %s", edit.edit_type)
    logging.info("Image path: %s", edit.image_path)
    logging.info("Category: %s", edit.category)
    logging.info("From attribute: %s", edit.from_attribute)
    logging.info("To attribute: %s", edit.to_attribute)
    logging.info("Prompt: %s", PreprocessingPipeline.generate_prompt(edit))

    # Get all inputs for the edit
    evaluation_input = evaluation_pipeline.get_all_inputs_for_edit(edit)

    # evaluate the edit using the corresponding operation
    # infer it from the edit type, so if edit type is "background" then use
    # Background() class from valuation/operations/background.py and so on
    # then call the evaluate_edit method of the class with the evaluation_input
    # as the argument
    # for example:
    if edit.edit_type.type_name == "object_addition":
        evaluation_output = ObjectAddition().evaluate_edit(
            evaluation_input,
        )
    elif edit.edit_type.type_name == "color":
        evaluation_output = ColorEdit().evaluate_edit(evaluation_input)
    elif edit.edit_type.type_name == "size":
        evaluation_output = SizeEdit().evaluate_edit(evaluation_input)

    # print the evaluation score if successful otherwise print evaluation failed
    if evaluation_output.success:
        if evaluation_output.score > 0:
            logging.info("Good sample!")
            logging.info(evaluation_output.score)
            logging.info(edit.image_path)
        logging.info(evaluation_output.score)
    else:
        logging.info("Evaluation failed")


if __name__ == "__main__":
    main()
