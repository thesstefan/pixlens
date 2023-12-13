import logging
import enum

import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from pixlens import utils
from pixlens.eval import sam as eval_sam

class OwlViTType(enum.StrEnum):
    base32 = "google/owlvit-base-patch32"
    base16 = "google/owlvit-base-patch16"
    large = "google/owlvit-large-patch14"
#They are automatically stored in users/$USER/.cache/huggingface/hub   

def load_owlvit(OwlViTType: OwlViTType.large , device: str = 'cpu'):
    processor = OwlViTProcessor.from_pretrained(OwlViTType)
    model = OwlViTForObjectDetection.from_pretrained(OwlViTType)
    model.to(device)
    model.eval()
    return model, processor





