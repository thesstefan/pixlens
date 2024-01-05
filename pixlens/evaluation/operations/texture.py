import torch

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces








class TextureEdit(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        to_attribute = evaluation_input.updated_strings.to_attribute