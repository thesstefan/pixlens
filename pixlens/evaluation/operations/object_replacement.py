import torch

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces
import numpy as np
from numpy import linalg as LA



class ObjectReplacement(evaluation_interfaces.OperationEvaluation):

    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        from_attribute = evaluation_input.updated_strings.from_attribute
        to_attribute = evaluation_input.updated_strings.to_attribute
        if to_attribute is None:
            return evaluation_interfaces.EvaluationOutput(
                score=0,
                success=False,
            )
        is_from_in_edited = (
            1
            if get_detection_segmentation_result_of_target(
                evaluation_input.edited_detection_segmentation_result,
                from_attribute,
            ).detection_output.phrases
            else 0
        )
        number_of_from_in_input = len(get_detection_segmentation_result_of_target(
            evaluation_input.input_detection_segmentation_result,
                from_attribute,
            ).detection_output.phrases)
        number_of_to_in_input = len(get_detection_segmentation_result_of_target(
            evaluation_input.input_detection_segmentation_result,
                to_attribute,
            ).detection_output.phrases)
        number_of_to_in_edited = len(get_detection_segmentation_result_of_target(
            evaluation_input.edited_detection_segmentation_result,
                to_attribute,
            ).detection_output.phrases)

        #calculate normalized distance
        norm_dist = 0

        #TODO(Ernesto): Extend this to the general case of more {from} and {to} objects present in category  # noqa: E501
        if number_of_from_in_input == 1 and number_of_to_in_input == 1 and number_of_to_in_edited == 1:  # noqa: E501
            #Measure distance between bounding boxes of to objects in edited image and from objects in original image  # noqa: E501
            bbox_from = get_detection_segmentation_result_of_target(
                evaluation_input.input_detection_segmentation_result,
                from_attribute,
            ).detection_output.bounding_boxes

            bbox_to = get_detection_segmentation_result_of_target(
                evaluation_input.edited_detection_segmentation_result,
                to_attribute,
            ).detection_output.bounding_boxes

            #compute normalized distance between bounding boxes
            from_center = np.array([(bbox_from[0] + bbox_from[2])/2, (bbox_from[1] + bbox_from[3])/2]) # noqa: E501
            to_center = np.array([(bbox_to[0] + bbox_to[2])/2, (bbox_to[1] + bbox_to[3])/2])

            #compute height of input image
            input_image_height = evaluation_input.input_image.height
            input_image_width = evaluation_input.input_image.width

            norm_dist = LA.norm(from_center - to_center)/(input_image_height + input_image_width)  # noqa: E501


        #if the number of to objects plus number of from object in original is different than number of to objects in edited return 0  # noqa: E501
        if number_of_from_in_input + number_of_to_in_input != number_of_to_in_edited:  # noqa: E501
            return evaluation_interfaces.EvaluationOutput(
                score=0,
                success=True,
            )

        #if from object still in image return score == 0
        if is_from_in_edited:
            return evaluation_interfaces.EvaluationOutput(
                score=0,
                success=True,
            )
        else:  # noqa: RET505
            return evaluation_interfaces.EvaluationOutput(
                score=1 - norm_dist,
                success=True,
            )


