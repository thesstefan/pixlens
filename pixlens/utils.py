import functools
import pathlib
import shutil
import typing

import platformdirs
import requests
import tqdm

CACHE_DIR_NAME = "pixlens"

T = typing.TypeVar("T")


def download_file(
    url: str,
    path: pathlib.Path,
    text: bool = False,
    desc: str = "",
) -> None:
    request = requests.get(url, stream=True, allow_redirects=True)

    if request.status_code != 200:
        request.raise_for_status()
        raise RuntimeError(
            f"Request to {url} returned status code {request.status_code}",
        )

    if text:
        with path.open("w") as file:
            file.write(request.text)

        return

    file_size = int(request.headers.get("Content-Length", 0))
    request.raw.head = functools.partial(request.raw.read, decode_content=True)

    with tqdm.tqdm.wrapattr(
        request.raw,
        "read",
        total=file_size,
        desc=desc,
    ) as request_raw:
        with path.open("wb") as file:
            shutil.copyfileobj(request_raw, file)


def get_cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path(platformdirs.user_cache_dir(CACHE_DIR_NAME))
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def get_basename_dict(path_dict: dict[T, str]) -> dict[T, str]:
    return {key: pathlib.Path(path).name for key, path in path_dict.items()}
