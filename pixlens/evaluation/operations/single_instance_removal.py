from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces


class SingleInstanceRemoval(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        category_in_input = get_detection_segmentation_result_of_target(
            evaluation_input.input_detection_segmentation_result,
            evaluation_input.updated_strings.category,
        )

        if len(category_in_input.detection_output.logits) == 0:
            return evaluation_interfaces.EvaluationOutput(
                success=False,
                edit_specific_score=0,
            )

        category_in_edited = get_detection_segmentation_result_of_target(
            evaluation_input.edited_detection_segmentation_result,
            evaluation_input.updated_strings.category,
        )

        num_categories_in_input = len(category_in_input.detection_output.logits)
        num_categories_in_edited = len(
            category_in_edited.detection_output.logits,
        )

        # if no change in number of categories, return score 0
        if num_categories_in_input <= num_categories_in_edited:
            return evaluation_interfaces.EvaluationOutput(
                success=True,
                edit_specific_score=0,
            )

        if num_categories_in_input - num_categories_in_edited == 1:
            # only one category removed
            return evaluation_interfaces.EvaluationOutput(
                success=True,
                edit_specific_score=1,
            )

        # more than one category removed
        return evaluation_interfaces.EvaluationOutput(
            success=True,
            edit_specific_score=0.5,
        )
        # TODO: maybe implement a more complex scoring system
        # where the score is inversely proportional to the number
        # of categories unnecessarily removed
