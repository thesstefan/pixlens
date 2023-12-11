import numpy as np
import torch
from groundingdino.util import box_ops, inference
from PIL import Image

from pixlens.eval import interfaces


# TODO: Decouple this from GroundingDINO, it should work for any bounding boxes.
def annotate_detection_output(
    image_source: np.ndarray, detection_output: interfaces.DetectionOutput
) -> Image.Image:
    height, width, _ = image_source.shape
    bbox = detection_output.bounding_boxes / torch.Tensor(
        [width, height, width, height]
    )
    bbox_cxcywh = box_ops.box_xyxy_to_cxcywh(bbox)

    annotated_frame = inference.annotate(
        image_source=image_source,
        boxes=bbox_cxcywh,
        logits=detection_output.logits,
        phrases=detection_output.phrases,
    )[..., ::-1]

    return Image.fromarray(annotated_frame)


def annotate_mask(mask: torch.Tensor, image: Image.Image) -> Image.Image:
    color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)

    height, width = mask.shape[-2:]
    mask_image = mask.cpu().reshape(height, width, 1) * color.reshape(1, 1, -1)

    mask_image_pil = Image.fromarray(
        (mask_image.numpy() * 255).astype(np.uint8)
    ).convert("RGBA")

    return Image.alpha_composite(image.convert("RGBA"), mask_image_pil)
