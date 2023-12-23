import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import torch

from pixlens.detection import grounded_sam, owl_vit_sam
from pixlens.detection import interfaces as detection_interfaces
from pixlens.editing import controlnet, pix2pix
from pixlens.editing import interfaces as editing_interfaces

T = TypeVar(
    "T",
    bound=detection_interfaces.PromptDetectAndBBoxSegmentModel
    | editing_interfaces.PromptableImageEditingModel,
)

P = ParamSpec("P")


def get_model(
    model_init: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    return model_init(*args, **kwargs)


@functools.cache
def get_detect_segment_model(
    model_type: str,
    device: torch.device,
) -> detection_interfaces.PromptDetectAndBBoxSegmentModel:
    match model_type:
        case "GroundedSAM":
            return get_model(grounded_sam.GroundedSAM, device=device)
        case "OwlViTSAM":
            return get_model(owl_vit_sam.OwlViTSAM, device=device)
        case _:
            raise NotImplementedError


@functools.cache
def get_edit_model(
    model_type: str,
    device: torch.device,
) -> editing_interfaces.PromptableImageEditingModel:
    match model_type:
        case "InstructPix2Pix":
            return get_model(
                pix2pix.Pix2pix,
                pix2pix_type=pix2pix.Pix2pixType.BASE,
                device=device,
            )
        case "ControlNet":
            return get_model(
                controlnet.ControlNet,
                pix2pix_type=controlnet.ControlNetType.BASE,
                device=device,
            )
        case _:
            raise NotImplementedError
