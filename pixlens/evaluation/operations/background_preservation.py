from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.operations import image_similarity

class ObjectRemoval(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        input_image = evaluation_input.input_image
        edited_image = evaluation_input.edited_image
