import logging
import pathlib

from pixlens.detection import grounded_sam, owl_vit_sam
from pixlens.detection.interfaces import PromptDetectAndBBoxSegmentModel
from pixlens.utils import yaml_constructible

logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(module)-30s | %(message)s",
    handlers=[logging.StreamHandler()],
)

NAME_TO_SEGMENT_DETECT_MODEL: dict[
    str,
    type[PromptDetectAndBBoxSegmentModel],
] = {
    "GroundedSAM": grounded_sam.GroundedSAM,
    "OwlViTSAM": owl_vit_sam.OwlViTSAM,
}


def load_detect_segment_model_from_yaml(
    yaml_path: pathlib.Path | str,
) -> PromptDetectAndBBoxSegmentModel:
    return yaml_constructible.load_class_from_yaml(
        yaml_path,
        NAME_TO_SEGMENT_DETECT_MODEL,
    )
