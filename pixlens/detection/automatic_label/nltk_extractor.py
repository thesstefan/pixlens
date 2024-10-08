import logging

import nltk
import torch
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from PIL import Image

from pixlens.detection.automatic_label.blip import Blip, BlipType
from pixlens.detection.automatic_label.interfaces import (
    CaptionIntoObjectsModel,
)
from pixlens.utils.utils import get_cache_dir


class NLTKObjectExtractor(CaptionIntoObjectsModel):
    def __init__(self) -> None:
        cache_dir = get_cache_dir()
        str_cache_dir = str(cache_dir)
        if str_cache_dir not in nltk.data.path:
            nltk.data.path.append(str_cache_dir)
        self.ensure_nltk_resources(str_cache_dir)

    @staticmethod
    def ensure_nltk_resources(str_cache_dir: str) -> None:
        try:
            nltk.data.find("tokenizers/punkt", str_cache_dir)
            nltk.data.find("taggers/averaged_perceptron_tagger", str_cache_dir)
        except LookupError:
            logging.info("Downloading NLTK resources...")
            nltk.download("punkt", download_dir=str_cache_dir)
            nltk.download(
                "averaged_perceptron_tagger",
                download_dir=str_cache_dir,
            )

    def extract_objects_from_caption(self, caption: str) -> list[str]:
        tokens = word_tokenize(caption)
        tagged = pos_tag(tokens)
        return [word for word, pos in tagged if pos.startswith("NN")]


class ImageToObjectsNLTK(Blip, NLTKObjectExtractor):
    def __init__(
        self,
        device: torch.device | None = None,
        blip_type: BlipType | None = None,
    ) -> None:
        blip_type = blip_type if blip_type is not None else BlipType.BLIP2
        Blip.__init__(self, device=device, blip_type=blip_type)
        NLTKObjectExtractor.__init__(self)

    def image_to_objects(self, image: Image.Image) -> list[str]:
        # Using image_caption from Blip
        caption = self.image_caption(image)
        return self.extract_objects_from_caption(caption)
