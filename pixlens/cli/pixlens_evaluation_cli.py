import argparse
import json
import logging
from pathlib import Path

import torch

from pixlens.dataset.editval import EditDataset, EditValDataset
from pixlens.detection import load_detect_segment_model_from_yaml
from pixlens.editing import load_editing_model_from_yaml
from pixlens.editing.interfaces import (
    PromptableImageEditingModel,
    ImageEditingPromptType,
)
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
    required=False,
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

parser.add_argument(
    "--run-evaluation-pipeline",
    action="store_true",
    help="When set to true, the entire evaluation pipeline will be run",
    required=False,
)


# check arguments function. Check that either edit type or edit id is provided
# and that the edit type is valid by importing the edit type class from
# pixlens/evaluation/interfaces.py and checking that the edit type is in the
# class
#
def check_args(args: argparse.Namespace) -> None:
    if args.run_evaluation_pipeline:
        return
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
    edit_dataset: EditDataset,
) -> list[Edit]:
    if args.run_evaluation_pipeline:
        return edit_dataset.get_all_edits()

    if args.edit_id is None:
        if args.edit_type is None:
            error_msg = "Either edit id or edit type must be provided"
            raise ValueError(error_msg)

        edits_with_type = edit_dataset.get_all_edits_with_type(
            args.edit_type,
        )

        return edits_with_type if args.do_all_edits else [edits_with_type[0]]

    return [edit_dataset.get_edit(args.edit_id)]


def evaluate_edits(
    edits: list[Edit],
    editing_models: list[PromptableImageEditingModel],
    operation_evaluators: dict[EditType, list[OperationEvaluation]],
    evaluation_pipeline: EvaluationPipeline,
) -> None:
    for editing_model in editing_models:
        model_evaluation_dir = (
            get_cache_dir() / editing_model.model_id / "evaluation"
        )
        logging.info("Evaluating model: %s", editing_model.model_id)
        for edit in edits:
            edit_dir = model_evaluation_dir / str(edit.edit_id)

            if edit.edit_type not in operation_evaluators:
                logging.warning(
                    "No operation evaluator implemented for edit type: %s",
                    edit.edit_type,
                )
                continue

            logging.info("Evaluating edit: %s", edit.edit_id)
            logging.info("Edit type: %s", edit.edit_type)
            logging.info("Image path: %s", edit.image_path)
            logging.info("Category: %s", edit.category)
            logging.info("From attribute: %s", edit.from_attribute)
            logging.info("To attribute: %s", edit.to_attribute)
            prompt = (
                edit.instruction_prompt
                if editing_model.prompt_type
                == ImageEditingPromptType.INSTRUCTION
                else edit.description_prompt
            )
            logging.info("Prompt: %s", prompt)

            evaluation_input = evaluation_pipeline.get_all_inputs_for_edit(
                edit,
                editing_model,
            )
            evaluation_outputs = [
                evaluator.evaluate_edit(evaluation_input)
                for evaluator in operation_evaluators[edit.edit_type]
            ]

            for output in evaluation_outputs:
                output.persist(edit_dir)

            # TODO: Remove these. It's just for convenience so that we
            # can see all items in one place.
            evaluation_input.input_image.save(
                edit_dir / Path(edit.image_path).name,
            )
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
                (
                    edit_dir / ("ANNOTATED_" + evaluation_input.prompt)
                ).with_suffix(
                    ".png",
                ),
            )

            # Save results in dataset
            evaluation_pipeline.update_evaluation_dataset(
                edit,
                editing_model.model_id,
                evaluation_outputs,
            )

            if all(output.success for output in evaluation_outputs):
                logging.info("Evaluation was successful")
            else:
                logging.info("Evaluation failed")

            logging.info("")

        # save CSV after each model, just in case
        evaluation_pipeline.save_evaluation_dataset()


def init_operation_evaluations() -> dict[EditType, list[OperationEvaluation]]:
    hist_cmp_method = HistogramComparisonMethod.CORRELATION
    color_hist_bins = 32
    color_smoothing_sigma = 5.0

    subject_preservation = SubjectPreservation(
        sift_min_matches=5,
        color_hist_bins=color_hist_bins,
        hist_cmp_method=hist_cmp_method,
        color_smoothing_sigma=color_smoothing_sigma,
    )
    background_preservation = BackgroundPreservation()
    return {
        EditType.COLOR: [
            ColorEdit(
                color_hist_bins=color_hist_bins,
                hist_cmp_method=hist_cmp_method,
                synthetic_sigma=75.0,
                smoothing_sigma=color_smoothing_sigma,
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


def load_editing_models(
    args: argparse.Namespace,
) -> list[PromptableImageEditingModel]:
    editing_models = []

    if args.run_evaluation_pipeline:
        all_models = [
            "model_cfgs/controlnet.yaml",
            "model_cfgs/lcm.yaml",
            "model_cfgs/instruct_pix2pix.yaml",
            # "model_cfgs/diffedit.yaml",
            # "model_cfgs/null_text_inversion.yaml",
        ]

        for model_yaml in all_models:
            editing_model = load_editing_model_from_yaml(model_yaml)
            editing_models.append(editing_model)
    else:
        editing_model = load_editing_model_from_yaml(args.editing_model_yaml)
        editing_models.append(editing_model)

    return editing_models


def postprocess_evaluation(
    editing_models: list[PromptableImageEditingModel],
    evaluation_pipeline: EvaluationPipeline,
) -> None:
    evaluation_pipeline.save_evaluation_dataset()

    results: dict[str, dict] = {}

    results["model_aggregation"] = {}
    for editing_model in editing_models:
        model_results = evaluation_pipeline.get_aggregated_scores_for_model(
            editing_model.model_id,
        )
        results["model_aggregation"][editing_model.model_id] = model_results

    results[
        "edit_type_aggregation"
    ] = evaluation_pipeline.get_aggregated_scores_for_edit_type()

    results_path = Path(get_cache_dir(), "evaluation_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as results_file:
        json.dump(results, results_file, indent=4)


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_args(args)

    edit_dataset = EditValDataset(
        Path("./pixlens/editval/object.json"),
        Path("./editval_instances/"),
    )
    preprocessing_pipe = PreprocessingPipeline(edit_dataset)
    editing_models = load_editing_models(args)
    preprocessing_pipe.execute_pipeline(models=editing_models)

    evaluation_pipeline = EvaluationPipeline(device=device)
    detection_model = load_detect_segment_model_from_yaml(
        args.detection_model_yaml,
    )
    evaluation_pipeline.init_editing_models(editing_models)
    evaluation_pipeline.init_detection_model(detection_model)

    edits = get_edits(args, edit_dataset)
    operation_evaluations = init_operation_evaluations()

    evaluate_edits(
        edits,
        editing_models,
        operation_evaluations,
        evaluation_pipeline,
    )

    # evaluation_pipeline.load_evaluation_dataset()
    postprocess_evaluation(editing_models, evaluation_pipeline)


if __name__ == "__main__":
    main()
