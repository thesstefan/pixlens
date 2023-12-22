import enum

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
)

from pixlens.detection.automatic_label.interfaces import (
    ImageCaption,
    ImageDescriptorModel,
)
from pixlens.detection.utils import log_if_hugging_face_model_not_in_cache
from pixlens.utils.utils import get_cache_dir


class BlipType(enum.StrEnum):
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    BLIP1 = "Salesforce/blip-image-captioning-base"


def load_blip(
    blip_type: BlipType,
    device: torch.device | None = None,
) -> tuple[
    Blip2ForConditionalGeneration | BlipForConditionalGeneration,
    Blip2Processor | AutoProcessor,
]:
    log_if_hugging_face_model_not_in_cache(blip_type)
    path_to_cache = get_cache_dir()
    model = (
        Blip2ForConditionalGeneration
        if blip_type in [BlipType.BLIP2]
        else BlipForConditionalGeneration
    ).from_pretrained(blip_type, cache_dir=path_to_cache)
    processor = (
        Blip2Processor if blip_type in [BlipType.BLIP2] else AutoProcessor
    ).from_pretrained(blip_type, cache_dir=path_to_cache)

    model.to(device)
    model.eval()
    return model, processor


class Blip(ImageDescriptorModel):
    device: torch.device | None
    model: torch.nn.Module

    def __init__(
        self,
        blip_type: BlipType = BlipType.BLIP2,
        device: torch.device | None = None,
    ) -> None:
        self.device = device
        self.model, self.processor = load_blip(blip_type, device)
        self.blip_type = blip_type

    def __call__(self, image: Image) -> ImageCaption:
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device, torch.float16
        )
        if self.blip_type == BlipType.BLIP2:
            generated_ids = self.model.generate(**inputs)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
        else:
            text = "A picture of"
            inputs = self.processor(
                images=image, text=text, return_tensors="pt"
            )
            generated_text = self.model(**inputs)
        return ImageCaption(caption=generated_text)


import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

blip = Blip()
print(blip(image))
