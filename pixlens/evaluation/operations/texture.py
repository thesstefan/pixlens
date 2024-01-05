import torch
import logging

from pixlens.detection.utils import get_detection_segmentation_result_of_target
from pixlens.evaluation import interfaces as evaluation_interfaces
from pixlens.detection import interfaces as detection_interfaces
from pixlens.detection.automatic_label import blip








class TextureEdit(evaluation_interfaces.OperationEvaluation):
    def evaluate_edit(
        self,
        evaluation_input: evaluation_interfaces.EvaluationInput,
    ) -> evaluation_interfaces.EvaluationOutput:
        to_attribute = evaluation_input.updated_strings.to_attribute
        
        category = evaluation_input.updated_strings.category
        input_segmentation = evaluation_input.input_detection_segmentation_result.segmentation_output
        input_detection = evaluation_input.input_detection_segmentation_result.detection_output
        edit_segmentation = evaluation_input.edited_detection_segmentation_result.segmentation_output
        edit_detection = evaluation_input.edited_detection_segmentation_result.detection_output

        score = 1

        ### Step 1: Check that the "category" object is in the input and edited image
        
        if not category in input_detection.phrases:
            logging.warning(
                "Texture edit could not be evaluated, because no object was "
                "present at input",
            )
            return evaluation_interfaces.EvaluationOutput(
                score=-1.0,
                success=False,
            )  # Object wasn't even present at input
        if not category in edit_detection.phrases:
            logging.warning(
                "Texture edit could not be evaluated, because no object was "
                "present at output",
            )
            return evaluation_interfaces.EvaluationOutput(
                score=0,
                success=True,
            ) # Object wasn't present at output
        
        ### Step 2: Check that there are the same number of objects in the input and edited image
        if len(input_detection.phrases) != len(edit_detection.phrases):
            logging.warning(
                "Number of detected objects in input and output image differ",
            )

            score -= 0.15

    

        ### Step 3: Check that there are same number of "category" objects in the input and edited image
        if input_detection.phrases.count(category) != edit_detection.phrases.count(category):
            logging.warning(
                "Number of detected category object in input and output image differ",
            )
            score -= 0.25


        ### Step 2: Check that the texture of the "category" object has changed in the edited image
        texture_change_of_category = False
        myblip = blip.Blip(blip_type=blip.BlipType.BLIPCAP)
        print("finished loading blip")
        Image = evaluation_input.edited_image   
        question = "Is the texture of the" + category + to_attribute +"?"
        answer = myblip.ask_blip(Image, question) 
        if answer == "yes":
            texture_change_to_attribute = True


        ### Step 3: Check that hte texture of the reamining objects has not been changed too
        texture_change_of_other_objects = False
        question = "How many objects have the texture" + to_attribute +"?"
        editedImage = evaluation_input.edited_image   
        originalImage = evaluation_input.input_image
        answer_edited = myblip.ask_blip(editedImage, question)
        answer_original = myblip.ask_blip(originalImage, question)
        str_to_int = {
            'zero': 0,
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            # Add more if needed
        }
        
        if isinstance(answer_edited, int):
        # It's an integer, do nothing
            pass
        else:
            # It's a string, convert it to an integer
            answer_edited = str_to_int.get(answer_edited, answer_edited)

        # Do the same for answer_original
        if isinstance(answer_original, int):
            pass
        else:
            answer_original = str_to_int.get(answer_original, answer_original)

        # Now you can subtract them
        try:
            score -= abs(int(answer_edited) - int(answer_original))*0.2
        except:
            print("blip didn't output a number")

        return evaluation_interfaces.EvaluationOutput(
            score=max(score, 0),
            success=True,
        )



    
