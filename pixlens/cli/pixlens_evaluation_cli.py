import argparse
import logging

import torch

from pixlens.detection.grounded_sam import GroundedSAM
from pixlens.detection.owl_vit_sam import OwlViTSAM
from pixlens.editing.controlnet import ControlNet
from pixlens.editing.instruct_pix2pix import InstructPix2Pix
from pixlens.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)
from pixlens.evaluation.interfaces import (
    Edit,
    EditType,
    EvaluationInput,
    EvaluationOutput,
)
from pixlens.evaluation.operations.color import ColorEdit
from pixlens.evaluation.operations.object_addition import ObjectAddition
from pixlens.evaluation.operations.object_removal import ObjectRemoval
from pixlens.evaluation.operations.object_replacement import ObjectReplacement
from pixlens.evaluation.operations.size import SizeEdit
from pixlens.evaluation.operations.texture import TextureEdit
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
parser.add_argument(
    "--do-all-edits",
    action="store_true",
    help="Evaluate all edits of the given edit type",
    required=False,
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
            EditType(args.edit_type)
        except ValueError as err:
            error_msg = "Invalid edit type provided"
            raise ValueError(error_msg) from err


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu") #TODO: remove this line

    check_args(args)

    preprocessing_pipe = PreprocessingPipeline(
        "./pixlens/editval/object.json",
        "./editval_instances/",
    )
    editing_model = get_editing_model(args.editing_model, device)
    preprocessing_pipe.execute_pipeline(models=[editing_model])

    evaluation_pipeline = EvaluationPipeline(device=device)
    detection_model = get_detection_model(args.detection_model, device)
    evaluation_pipeline.init_editing_model(editing_model)
    evaluation_pipeline.init_detection_model(detection_model)

    edits = get_edits(args, preprocessing_pipe, evaluation_pipeline)

    overall_score, successful_edits = evaluate_edits(
        edits,
        editing_model,
        evaluation_pipeline,
    )

    logging.info("Overall score: %s", overall_score / successful_edits)
    logging.info(
        "Percentage of successful edits: %s",
        successful_edits / len(edits),
    )


def get_editing_model(
    model_name: str,
    device: torch.device,
) -> ControlNet | InstructPix2Pix:
    if model_name.lower() == "controlnet":
        return ControlNet(device=device)
    if model_name.lower() == "pix2pix":
        return InstructPix2Pix(device=device)
    error_msg = f"Invalid editing model name: {model_name.lower()}"
    raise ValueError(error_msg)


def get_detection_model(
    model_name: str,
    device: torch.device,
) -> GroundedSAM | OwlViTSAM:
    if model_name.lower() == "groundedsam":
        return GroundedSAM(device=device, detection_confidence_threshold=0.5)
    if model_name.lower() == "owlvitsam":
        return OwlViTSAM(device=device)
    error_msg = "Invalid detection model name"
    raise ValueError(error_msg)


def get_edits(
    args: argparse.Namespace,
    preprocessing_pipe: PreprocessingPipeline,
    evaluation_pipeline: EvaluationPipeline,
) -> list[Edit]:
    if args.edit_id is None:
        all_edits_by_type = preprocessing_pipe.get_all_edits_edit_type(
            args.edit_type,
        )
        if not args.do_all_edits:
            random_edit_record = all_edits_by_type.iloc[[1]]
            edit = preprocessing_pipe.get_edit(
                random_edit_record["edit_id"].astype(int).to_numpy()[0],
                evaluation_pipeline.edit_dataset,
            )
            edits = [edit]
        else:
            edits = []
            for i in range(len(all_edits_by_type)):
                random_edit_record = all_edits_by_type.iloc[[i]]
                edit = preprocessing_pipe.get_edit(
                    random_edit_record["edit_id"].astype(int).to_numpy()[0],
                    evaluation_pipeline.edit_dataset,
                )
                edits.append(edit)
    else:
        edit = preprocessing_pipe.get_edit(
            args.edit_id,
            evaluation_pipeline.edit_dataset,
        )
        edits = [edit]
    return edits


def evaluate_edits(
    edits: list[Edit],
    editing_model: ControlNet | InstructPix2Pix,
    evaluation_pipeline: EvaluationPipeline,
) -> tuple[float, int]:
    overall_score = 0.0
    successful_edits = 0
    for edit in edits:
        logging.info("Running edit: %s", edit.edit_id)
        logging.info("Edit type: %s", edit.edit_type)
        logging.info("Image path: %s", edit.image_path)
        logging.info("Category: %s", edit.category)
        logging.info("From attribute: %s", edit.from_attribute)
        logging.info("To attribute: %s", edit.to_attribute)
        logging.info("Prompt: %s", editing_model.generate_prompt(edit))

        evaluation_input = evaluation_pipeline.get_all_inputs_for_edit(edit)

        evaluation_output = evaluate_edit(edit, evaluation_input)

        if evaluation_output.success:
            successful_edits += 1
            overall_score += evaluation_output.edit_specific_score
            if evaluation_output.edit_specific_score > 0:
                logging.info("Good sample!")
                logging.info(evaluation_output.edit_specific_score)
                logging.info(edit.image_path)
            logging.info(evaluation_output.edit_specific_score)
        else:
            logging.info("Evaluation failed")

    return overall_score, successful_edits


def evaluate_edit(
    edit: Edit,
    evaluation_input: EvaluationInput,
) -> EvaluationOutput:
    if edit.edit_type == EditType.OBJECT_ADDITION:
        return ObjectAddition().evaluate_edit(evaluation_input)
    if edit.edit_type == EditType.COLOR:
        return ColorEdit().evaluate_edit(evaluation_input)
    if edit.edit_type == EditType.SIZE:
        return SizeEdit().evaluate_edit(evaluation_input)
    if edit.edit_type == EditType.OBJECT_REMOVAL:
        return ObjectRemoval().evaluate_edit(evaluation_input)
    if edit.edit_type == EditType.OBJECT_REPLACEMENT:
        return ObjectReplacement().evaluate_edit(evaluation_input)
    if edit.edit_type == EditType.TEXTURE:
        return TextureEdit().evaluate_edit(evaluation_input)
    error_msg = f"Invalid edit type: {edit.edit_type}"
    raise ValueError(error_msg)


if __name__ == "__main__":
    main()
