import functools

import torch

from pixlens.eval import grounded_sam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO(thesstefan): Update this to work with any model
@functools.cache
def get_detect_and_segment_model() -> grounded_sam.GroundedSAM:
    return grounded_sam.GroundedSAM(device=device)
