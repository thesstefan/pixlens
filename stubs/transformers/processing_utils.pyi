# ruff: noqa: FBT001, PLR0913

import os
from typing import Self, Any

from _typeshed import Incomplete

class ProcessorMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[Any],
        cache_dir: str | os.PathLike[Any] | None = ...,
        force_download: bool = ...,
        local_files_only: bool = ...,
        token: str | bool | None = ...,
        revision: str = ...,
        **kwargs: Incomplete,
    ) -> Self: ...
