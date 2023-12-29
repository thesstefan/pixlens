import argparse
import logging

import requests
import torch
from PIL import Image

from pixlens.detection.automatic_label import blip, nltk_extractor

parser = argparse.ArgumentParser(
    description="BLIP Image Captioning - Generate captions for images",
)
parser.add_argument("--model", type=str, default="blip1", help="Model to use")


def main() -> None:
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing %s model", args.model)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)  # noqa: S113
    if args.model == "blip1":
        bliptype = blip.BlipType.BLIP1
    elif args.model == "blip2":
        bliptype = blip.BlipType.BLIP2
    else:
        msg = "Invalid model"
        raise ValueError(msg)
    image_to_objects = nltk_extractor.ImageToObjectsNLTK(
        device=torch.device("cpu"),
        blip_type=bliptype,
    )
    caption = image_to_objects.image_to_objects(image)
    logging.info("Objects: %s", caption)


if __name__ == "__main__":
    main()
