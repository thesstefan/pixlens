import functools
import hashlib
import logging
import pathlib
import shutil
import typing

import PIL
import platformdirs
import requests
import tqdm
from PIL import Image

CACHE_DIR_NAME = "pixlens"
REQUEST_TIMEOUT = 10

T = typing.TypeVar("T")


# TODO(thesstefan): I don't like this. Get rid of it sometime.
class GotNoneError(ValueError):
    """Use when required optional doesn't have a value."""

    def __init__(self, msg: str) -> None:
        super().__init__(f"{msg}. Got None.")


def _request_file(url: str) -> requests.Response:
    response = requests.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        response.raise_for_status()

    return response


def download_text_file(
    url: str,
    path: pathlib.Path,
) -> None:
    response = _request_file(url)

    with path.open("w") as file:
        file.write(response.text)


def download_file(
    url: str,
    path: pathlib.Path,
    desc: str = "",
) -> None:
    response = _request_file(url)
    file_size = int(response.headers.get("Content-Length", 0))
    response.raw.head = functools.partial(
        response.raw.read,
        decode_content=True,
    )

    with tqdm.tqdm.wrapattr(
        response.raw,
        "read",
        total=file_size,
        desc=desc,
    ) as request_raw, path.open("wb") as file:
        shutil.copyfileobj(request_raw, file)


def get_cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path(platformdirs.user_cache_dir(CACHE_DIR_NAME))
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def get_basename_dict(path_dict: dict[T, str]) -> dict[T, str]:
    return {key: pathlib.Path(path).name for key, path in path_dict.items()}


def download_image(url: str) -> Image.Image:
    image = PIL.Image.open(
        requests.get(url, stream=True, timeout=REQUEST_TIMEOUT).raw,
    )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def log_if_hugging_face_model_not_in_cache(
    model_type: str,
    cache_dir: pathlib.Path | None = None,
) -> None:
    if cache_dir is None:
        cache_dir = get_cache_dir()
    model_dir = model_type.replace("/", "--")
    model_dir = "models--" + model_dir

    if not (cache_dir / model_dir).is_dir():
        logging.info("Downloading model %s...", model_type)


def get_image_extension(
    image_path: pathlib.Path,
) -> str | None:
    # If there is already an extension, return it
    if image_path.suffix:
        return image_path.suffix

    # If not, try to infer the extension
    for ext in [".jpg", ".png", ".jpeg", ".bmp", ".gif", ".tiff"]:
        if image_path.with_suffix(ext).is_file():
            return ext
    return None


def hash_primitive_list(
    items: list[str | bool | int | float],
    hash_length: int = 3,
) -> str:
    item_str = "".join(map(str, items))

    return hashlib.shake_256(item_str.encode()).hexdigest(hash_length)
