import enum
import logging
import pathlib

import numpy as np
import segment_anything
import torch
from PIL import Image

from pixlens.detection import interfaces
from pixlens.utils import utils


class SAMType(enum.StrEnum):
    VIT_H = "vit_h"
    VIT_L = "vit_l"
    VIT_B = "vit_b"


SAM_CKPT_URLS = {
    SAMType.VIT_H: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    SAMType.VIT_L: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    SAMType.VIT_B: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

SAM_CKPT_NAMES = {
    sam_type: pathlib.Path(ckpt_url).name
    for sam_type, ckpt_url in SAM_CKPT_URLS.items()
}


def get_sam_ckpt(sam_type: SAMType) -> pathlib.Path:
    ckpt_path = utils.get_cache_dir() / SAM_CKPT_NAMES[sam_type]

    if not ckpt_path.exists():
        logging.info(
            "Downloading SAM %s weights from %s...",
            sam_type,
            SAM_CKPT_URLS[sam_type],
        )
        utils.download_file(
            SAM_CKPT_URLS[sam_type],
            ckpt_path,
            desc=f"SAM {sam_type}",
        )

    return ckpt_path


def load_sam_predictor(
    sam_type: SAMType = SAMType.VIT_H,
    device: torch.device | None = None,
) -> segment_anything.SamPredictor:
    sam_ckpt = get_sam_ckpt(sam_type)

    logging.info("Loading SAM %s from %s...", sam_type, sam_ckpt)

    sam = segment_anything.sam_model_registry[sam_type](checkpoint=sam_ckpt).to(
        device,
    )

    return segment_anything.SamPredictor(sam)


class BBoxSamPredictor(interfaces.BBoxSegmentationModel):
    sam_predictor: segment_anything.SamPredictor

    def __init__(
        self,
        sam_type: SAMType = SAMType.VIT_H,
        device: torch.device | None = None,
    ) -> None:
        self.sam_predictor = load_sam_predictor(sam_type, device)

    def segment(
        self,
        bbox: torch.Tensor,
        image: Image.Image,
    ) -> interfaces.SegmentationOutput:
        image_array = np.asarray(image)

        self.sam_predictor.set_image(image_array)

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            bbox,
            image_array.shape[:2],
        ).to(self.sam_predictor.device)

        masks, logits, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            # TODO: These could be useful
            return_logits=False,
            multimask_output=False,
        )

        return interfaces.SegmentationOutput(masks, logits)

    @property
    def device(self) -> torch.device:
        return self.sam_predictor.device
