import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from pixlens.detection.automatic_label.interfaces import CaptionIntoObjects
from pixlens.utils.utils import get_cache_dir


class NLTKObjectExtractor(CaptionIntoObjects):
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
            print("Downloading NLTK resources...")
            nltk.download("punkt", download_dir=str_cache_dir)
            nltk.download(
                "averaged_perceptron_tagger", download_dir=str_cache_dir
            )

    def extract_objects(self, caption: str) -> list[str]:
        tokens = word_tokenize(caption)
        tagged = pos_tag(tokens)
        return [word for word, pos in tagged if pos.startswith("NN")]
