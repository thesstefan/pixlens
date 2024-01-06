from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.operations import image_similarity


class BackgroundPreservation(evaluation_interfaces.GeneralEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> float:
        input_image = evaluation_input.input_image
        edited_image = evaluation_input.edited_image

        return 0.0
