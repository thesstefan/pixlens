import logging
import enum
from pathlib import Path
import os
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)

from pixlens import utils


class OwlViTType(enum.StrEnum):
    BASE32 = "google/owlvit-base-patch32"
    BASE16 = "google/owlvit-base-patch16"
    LARGE = "google/owlvit-large-patch14"
    BASE16_V2 = "google/owlv2-base-patch16"
    LARGE_V2 = "google/owlv2-large-patch14"


# They are automatically stored in users/$USER/.cache/huggingface/hub


def log_if_model_not_in_cache(model_name: str, cache_dir: Path) -> None:
    folder_name = model_name.replace("/", "--")
    folder_name = "models--" + folder_name
    # Construct the full path to the model folder within the cache
    full_path = cache_dir / folder_name
    # Check if the folder exists
    if not full_path.is_dir():
        logging.info(f"Downloading OwlViT model from {model_name}...")


def load_owlvit(owlvit_type: OwlViTType, device: str = "cpu"):
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
