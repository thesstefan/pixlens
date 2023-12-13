import logging

import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection


from pixlens.eval import sam as eval_sam

def load_owlvit(ckpt_path: str, device: str = 'cpu'):
    processor = OwlViTProcessor.from_pretrained(ckpt_path)
    model = OwlViTForObjectDetection.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()
    return model, processor

