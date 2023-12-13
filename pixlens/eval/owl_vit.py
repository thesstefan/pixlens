import logging
import enum
from pathlib import Path
import os
from transformers import OwlViTProcessor, OwlViTForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection

from pixlens import utils

class OwlViTType(enum.StrEnum):
    base32 = "google/owlvit-base-patch32"
    base16 = "google/owlvit-base-patch16"
    large = "google/owlvit-large-patch14"
    base16v2="google/owlv2-base-patch16"
    largev2="google/owlv2-large-patch14"
#They are automatically stored in users/$USER/.cache/huggingface/hub   

def log_if_model_not_in_cache(model_name: str, cache_dir: Path) -> None:
    folder_name = model_name.replace('/', '--')
    folder_name = "models--" + folder_name
    # Construct the full path to the model folder within the cache
    full_path = cache_dir / folder_name
    # Check if the folder exists
    if not full_path.is_dir():
        logging.info(f"Downloading OwlViT model from {model_name}...")

def load_owlvit(OwlViTType: OwlViTType.large , device: str = 'cpu'):
    path_to_cache = utils.get_cache_dir()
    log_if_model_not_in_cache(OwlViTType, path_to_cache)
    if 'v2' in OwlViTType:
        processor = Owlv2Processor.from_pretrained(OwlViTType, cache_dir=path_to_cache)
        model = Owlv2ForObjectDetection.from_pretrained(OwlViTType, cache_dir=path_to_cache)
    else:
        processor = OwlViTProcessor.from_pretrained(OwlViTType, cache_dir=path_to_cache)
        model = OwlViTForObjectDetection.from_pretrained(OwlViTType, cache_dir=path_to_cache)
    model.to(device)
    model.eval()
    return model, processor


