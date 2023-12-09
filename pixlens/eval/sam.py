import enum
import logging
import pathlib

import segment_anything

from pixlens import utils


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
            f"Downloading SAM {sam_type} weights from {SAM_CKPT_URLS[sam_type]}..."
        )
        utils.download_file(SAM_CKPT_URLS[sam_type], ckpt_path, desc=f"SAM {sam_type}")

    return ckpt_path


def load_sam_predictor(sam_type: SAMType = SAMType.VIT_H):
    sam_ckpt = get_sam_ckpt(sam_type)

    logging.info(f"Loading SAM {sam_type} from {sam_ckpt}...")

    sam = segment_anything.sam_model_registry[sam_type](checkpoint=sam_ckpt)
    predictor = segment_anything.SamPredictor(sam)

    return predictor
