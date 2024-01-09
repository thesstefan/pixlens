from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces
from skimage.metrics import structural_similarity
import cv2
import numpy as np


class BackgroundPreservation(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:

        category = evaluation_input.updated_strings.category
        input_segmentation = evaluation_input.input_detection_segmentation_result.segmentation_output
        input_detection = evaluation_input.input_detection_segmentation_result.detection_output
        edit_segmentation = evaluation_input.edited_detection_segmentation_result.segmentation_output
        edit_detection = evaluation_input.edited_detection_segmentation_result.detection_output

        score = 1.0
        

        return 0
            
