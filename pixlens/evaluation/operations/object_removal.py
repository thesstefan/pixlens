import torch

from pixlens.evaluation import interfaces as evaluation_interfaces


class ObjectAddition(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        raise NotImplementedError
    
    def is_object_in_input_and_not_in_output(evaluation_interfaces.EvaluationInput) -> tuple[bool, bool]:
        raise NotImplementedError