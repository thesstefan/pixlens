import torch

from pixlens.evaluation import interfaces as evaluation_interfaces


class ObjectAddition(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        input_segmentation = evaluation_input.input_detection_segmentation_result.segmentation_output
        edit_segmentation = evaluation_input.edited_detection_segmentation_result.segmentation_output
        input_detection = evaluation_input.input_detection_segmentation_result.detection_output
        edit_detection = evaluation_input.edited_detection_segmentation_result.detection_output

        return evaluation_interfaces.EvaluationOutput(score=0.0)
