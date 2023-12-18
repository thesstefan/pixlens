import logging
import enum
from pathlib import Path

from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)
import torch
from PIL import Image

from pixlens.utils import utils
from pixlens.utils import interfaces


class OwlViTType(enum.StrEnum):
    BASE32 = "google/owlvit-base-patch32"
    BASE16 = "google/owlvit-base-patch16"
    LARGE = "google/owlvit-large-patch14"
    BASE16_V2 = "google/owlv2-base-patch16"
    LARGE_V2 = "google/owlv2-large-patch14"


# They are automatically stored in users/$USER/.cache/huggingface/hub


def log_if_model_not_in_cache(model_name: str, cache_dir: Path) -> None:
    model_dir = model_name.replace("/", "--")
    model_dir = "models--" + model_dir
    # Construct the full path to the model folder within the cache
    full_path = cache_dir / model_dir
    # Check if the folder exists
    if not full_path.is_dir():
        logging.info(f"Downloading OwlViT model from {model_name}...")


def load_owlvit(owlvit_type: OwlViTType, device: torch.device | None = None):
    path_to_cache = utils.get_cache_dir()
    log_if_model_not_in_cache(owlvit_type, path_to_cache)
    if owlvit_type in [OwlViTType.BASE16_V2, OwlViTType.LARGE_V2]:
        processor = Owlv2Processor.from_pretrained(owlvit_type, cache_dir=path_to_cache)
        model = Owlv2ForObjectDetection.from_pretrained(
            owlvit_type, cache_dir=path_to_cache
        )
    else:
        processor = OwlViTProcessor.from_pretrained(
            owlvit_type, cache_dir=path_to_cache
        )
        model = OwlViTForObjectDetection.from_pretrained(
            owlvit_type, cache_dir=path_to_cache
        )
    model.to(device)
    model.eval()
    return model, processor


class OwLViT(interfaces.PromptableDetectionModel):
    device: torch.device | None

    def __init__(
        self,
        owlvit_type: OwlViTType,
        device: torch.device | None = None,
        detection_confidence_threshold: float = 0.3,
    ) -> None:
        self.device = device
        self.model, self.processor = load_owlvit(owlvit_type, device)
        self.detection_confidence_threshold = detection_confidence_threshold

    def output_into_detection_output(
        self, owlvit_results: list[dict], prompt: list[str]
    ) -> list:
        results_new = []
        for result in owlvit_results:
            scores = result["scores"]
            labels = [prompt[id] for id in result["labels"].tolist()]
            boxes = result["boxes"]

            detection_output = interfaces.DetectionOutput(
                bounding_boxes=boxes, logits=scores, phrases=labels
            )
            results_new.append(detection_output)
        return results_new

    def detect(self, prompt: str, image_path: str) -> list:
        image = Image.open(image_path)
        prompts = prompt.split(",")
        inputs = self.processor(text=prompts, images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.detection_confidence_threshold,
            target_sizes=torch.Tensor([image.size[::-1]]).to(self.device),
        )
        results = self.output_into_detection_output(results, prompts)

        # Move all tensors to CPU
        for result in results:
            result.bounding_boxes = result.bounding_boxes.cpu()
            result.logits = result.logits.cpu()

        return results[0]
