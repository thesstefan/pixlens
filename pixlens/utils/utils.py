import functools
import pathlib
import shutil
import typing
import platformdirs
import requests
import tqdm
import PIL
from PIL import Image

CACHE_DIR_NAME = "pixlens"
REQUEST_TIMEOUT = 10

T = typing.TypeVar("T")


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


def download_image(url) -> Image.Image:
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
