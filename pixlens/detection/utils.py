import logging
from pathlib import Path

from pixlens.utils import utils


def log_if_hugging_face_model_not_in_cache(
    model_type: str, cache_dir: Path | None = None
) -> None:
    if cache_dir is None:
        cache_dir = utils.get_cache_dir()
    model_dir = model_type.replace("/", "--")
    model_dir = "models--" + model_dir

    if not (cache_dir / model_dir).is_dir():
        logging.info("Downloading model %s...", model_type)
