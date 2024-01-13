import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pixlens.detection import load_detect_segment_model_from_yaml
from pixlens.editing import load_editing_model_from_yaml
from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)
from pixlens.evaluation.interfaces import (
    Edit,
    EditType,
    OperationEvaluation,
)
from pixlens.evaluation.operations.alter_parts import AlterParts
from pixlens.evaluation.operations.background_preservation import (
    BackgroundPreservation,
)
from pixlens.evaluation.operations.color import ColorEdit
from pixlens.evaluation.operations.object_addition import ObjectAddition
from pixlens.evaluation.operations.object_removal import ObjectRemoval
from pixlens.evaluation.operations.object_replacement import ObjectReplacement
from pixlens.evaluation.operations.position_replacement import (
    PositionReplacement,
)
from pixlens.evaluation.operations.positional_addition import PositionalAddition
from pixlens.evaluation.operations.size import SizeEdit
from pixlens.evaluation.operations.subject_preservation import (
    SubjectPreservation,
)
from pixlens.evaluation.preprocessing_pipeline import PreprocessingPipeline
from pixlens.evaluation.utils import HistogramComparisonMethod
from pixlens.utils.utils import get_cache_dir

parser = argparse.ArgumentParser(description="Evaluate PixLens Editing Model")
parser.add_argument(
    "--edit-id",
    type=int,
    required=False,
    help="ID of the edit to be evaluated",
)
parser.add_argument(
    "--editing-model-yaml",
    type=str,
    required=True,
    help="YAML configuration of the editing model to use",
)
parser.add_argument(
    "--detection-model-yaml",
    type=str,
    required=True,
    help="YAML configuration of the detection model to use",
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
    editing_model: PromptableImageEditingModel,
    operation_evaluators: dict[EditType, list[OperationEvaluation]],
    evaluation_pipeline: EvaluationPipeline,
) -> tuple[float, int]:
    overall_score = 0.0
    successful_edits = 0
    model_evaluation_dir = (
        get_cache_dir() / editing_model.model_id / "evaluation"
    )
    for edit in edits:
        edit_dir = model_evaluation_dir / str(edit.edit_id)

        logging.info("Evaluating edit: %s", edit.edit_id)
        logging.info("Edit type: %s", edit.edit_type)
        logging.info("Image path: %s", edit.image_path)
        logging.info("Category: %s", edit.category)
        logging.info("From attribute: %s", edit.from_attribute)
        logging.info("To attribute: %s", edit.to_attribute)
        logging.info("Prompt: %s", editing_model.generate_prompt(edit))

        evaluation_input = evaluation_pipeline.get_all_inputs_for_edit(edit)
        evaluation_outputs = [
            evaluator.evaluate_edit(evaluation_input)
            for evaluator in operation_evaluators[edit.edit_type]
        ]

        for output in evaluation_outputs:
            output.persist(edit_dir)

        # TODO: Remove these. It's just for convenience so that we
        # can see all items in one place.
        evaluation_input.input_image.save(edit_dir / Path(edit.image_path).name)
        evaluation_input.edited_image.save(
            (edit_dir / evaluation_input.prompt).with_suffix(".png"),
        )
        evaluation_input.annotated_input_image.save(
            Path(
                (
                    edit_dir / ("ANNOTATED_" + Path(edit.image_path).name)
                ).with_suffix(""),
            ).with_suffix(".png"),
        )
        evaluation_input.annotated_edited_image.save(
            (edit_dir / ("ANNOTATED_" + evaluation_input.prompt)).with_suffix(
                ".png",
            ),
        )

        if all(output.success for output in evaluation_outputs):
            successful_edits += 1
            logging.info("Evaluation was successful")
        else:
            logging.info("Evaluation failed")

        logging.info("")

    return overall_score, successful_edits


def init_operation_evaluations() -> dict[EditType, list[OperationEvaluation]]:
    hist_cmp_method = HistogramComparisonMethod.CORRELATION
    color_hist_bins = 32

    subject_preservation = SubjectPreservation(
        sift_min_matches=5,
        color_hist_bins=color_hist_bins,
        hist_cmp_method=hist_cmp_method,
    )
    background_preservation = BackgroundPreservation()
    return {
        EditType.COLOR: [
            ColorEdit(
                color_hist_bins=color_hist_bins,
                hist_cmp_method=hist_cmp_method,
                synthetic_sigma=75.0,
            ),
            subject_preservation,
            background_preservation,
        ],
        EditType.SIZE: [
            SizeEdit(),
            subject_preservation,
            background_preservation,
        ],
        EditType.POSITION_REPLACEMENT: [
            PositionReplacement(),
            subject_preservation,
            background_preservation,
        ],
        EditType.POSITIONAL_ADDITION: [
            PositionalAddition(),
            background_preservation,
        ],
        EditType.OBJECT_ADDITION: [
            ObjectAddition(),
            background_preservation,
        ],
        EditType.OBJECT_REMOVAL: [
            ObjectRemoval(),
            background_preservation,
        ],
        EditType.OBJECT_REPLACEMENT: [
            ObjectReplacement(),
            background_preservation,
        ],
        EditType.ALTER_PARTS: [
            AlterParts(),
            background_preservation,
        ],
    }


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_args(args)

    preprocessing_pipe = PreprocessingPipeline(
        "./pixlens/editval/object.json",
        "./editval_instances/",
    )
    editing_model = load_editing_model_from_yaml(args.editing_model_yaml)
    preprocessing_pipe.execute_pipeline(models=[editing_model])

    evaluation_pipeline = EvaluationPipeline(device=device)
    detection_model = load_detect_segment_model_from_yaml(
        args.detection_model_yaml,
    )
    evaluation_pipeline.init_editing_model(editing_model)
    evaluation_pipeline.init_detection_model(detection_model)

    edits = get_edits(args, preprocessing_pipe, evaluation_pipeline)
    operation_evaluations = init_operation_evaluations()

    overall_score, successful_edits = evaluate_edits(
        edits,
        editing_model,
        operation_evaluations,
        evaluation_pipeline,
    )

    logging.info(
        "Percentage of successful edits: %s",
        successful_edits / len(edits),
    )


if __name__ == "__main__":
    main()
