import torch

from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.operations import image_similarity


class BackgroundPreservation(evaluation_interfaces.GeneralEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> float:
        input_image = evaluation_input.input_image
        edited_image = evaluation_input.edited_image
        mask_input, mask_edit = self.get_masks(evaluation_input)
        return image_similarity.masked_image_similarity(
            input_image,
            mask_input,
            edited_image,
            mask_edit,
        )

    def get_masks(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
