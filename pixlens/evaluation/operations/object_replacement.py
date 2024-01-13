import logging

from torchvision.ops import box_iou

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces


class ObjectReplacement(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        from_attribute = evaluation_input.updated_strings.from_attribute
        to_attribute = evaluation_input.updated_strings.to_attribute
        if to_attribute is None:
            logging.warning(
                "No {to} attribute provided in an "
                "object replacement operation.",
            )
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0,
                success=False,
            )
        if from_attribute is None:
            logging.warning(
                "No {from} attribute provided in an "
                "object replacement operation.",
            )
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0,
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

        if len(froms_in_input.detection_output.phrases) == 0:
            return evaluation_interfaces.EvaluationOutput(
                edit_specific_score=0,
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
            froms_in_input.detection_output.bounding_boxes,
        ):
            from_object_bbox = froms_in_input.detection_output.bounding_boxes[
                from_object_index
            ]
            max_iou = 0.0
            max_iou_index = -1

            iou = box_iou(
                from_object_bbox.unsqueeze(0),
                tos_in_edited.detection_output.bounding_boxes,
            )[0]

            if iou.max() > max_iou:
                max_iou = iou.max().item()
                max_iou_index = iou.argmax().item()

            # we could compare here iou with a threshold to make
            # it more robust but for the moment as max_iou
            # is initialized with 0.0, it will work kind of as
            # an implicit threshold, by checking that at least
            # the {to} object covers the {from} object

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
        false_positives = len(
            tos_in_edited.detection_output.bounding_boxes,
        ) - len(
            used_tos_in_edited,
        )

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        tol = 1e-06
        if precision + recall < tol:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        return evaluation_interfaces.EvaluationOutput(
            edit_specific_score=f1_score,
            success=True,
        )
