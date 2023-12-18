import logging
from pathlib import Path


def log_model_if_not_in_cache(model_name: str, cache_dir: Path) -> None:
    model_dir = model_name.replace("/", "--")
    model_dir = "models--" + model_dir
    full_path = cache_dir / model_dir
    if not full_path.is_dir():
        logging.info(f"Downloading model from {model_name}...")
