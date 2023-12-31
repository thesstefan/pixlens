import logging

import numpy as np
from numpy import linalg as LA

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.evaluation.utils import compute_bbox_iou


class ObjectReplacement(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        from_attribute = evaluation_input.updated_strings.from_attribute
        to_attribute = evaluation_input.updated_strings.to_attribute
        if to_attribute is None:
            logging.warning(
                "No {to} attribute provided in an object replacement operation.",
            )
            return evaluation_interfaces.EvaluationOutput(
                score=0,
                success=False,
            )
        if from_attribute is None:
            logging.warning(
                "No {from} attribute provided in an object replacement operation.",
            )
            return evaluation_interfaces.EvaluationOutput(
                score=0,
                success=False,
            )

        froms_in_input = get_detection_segmentation_result_of_target(
            evaluation_input.input_detection_segmentation_result,
            from_attribute,
        )
        tos_in_edited = get_detection_segmentation_result_of_target(
            evaluation_input.edited_detection_segmentation_result,
            to_attribute,
        )

        # if no from in input detected, then return 0
        if len(froms_in_input.detection_output.phrases) == 0:
            return evaluation_interfaces.EvaluationOutput(
                score=0,
                success=False,
            )

        # For the moment we don't consider the following:
        # froms_in_edited = get_detection_segmentation_result_of_target(
        #     evaluation_input.edited_detection_segmentation_result,
        #     from_attribute,
        # )
        # tos_in_input = get_detection_segmentation_result_of_target(
        #     evaluation_input.input_detection_segmentation_result,
        #     to_attribute,
        # )

        used_tos_in_edited = set()
        true_positives = 0
        false_negatives = 0

        # for each from object in input image, check for a corresponding to object with the highest IoU in edited image  # noqa: E501
        # that is not used yet
        for from_object_index, _ in enumerate(
            froms_in_input.detection_output.phrases,
        ):
            from_object_bbox = froms_in_input.detection_output.bounding_boxes[
                from_object_index
            ]
            max_iou = 0.0
            max_iou_index = -1
            for to_object_index, _ in enumerate(
                tos_in_edited.detection_output.phrases,
            ):
                if to_object_index in used_tos_in_edited:
                    continue
                to_object_bbox = tos_in_edited.detection_output.bounding_boxes[
                    to_object_index
                ]
                iou = compute_bbox_iou(from_object_bbox, to_object_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = to_object_index

            if max_iou_index != -1:
                # detected a corresponding to object that
                # "covers" and replaces the from object
                true_positives += 1
                used_tos_in_edited.add(max_iou_index)
            else:
                # no corresponding to object found
                false_negatives += 1

        # compute the number of to objects that are not used
        # in the edited image
        false_positives = len(tos_in_edited.detection_output.phrases) - len(
            used_tos_in_edited,
        )

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        f1_score = 2 * (precision * recall) / (precision + recall)
        return evaluation_interfaces.EvaluationOutput(
            score=f1_score,
            success=True,
        )
