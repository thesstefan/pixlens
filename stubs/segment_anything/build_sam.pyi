from pathlib import Path
from typing import Protocol

from .modeling import Sam as Sam

class SamBuilder(Protocol):
    def __call__(self, checkpoint: str | Path | None = ...) -> Sam: ...

def build_sam_vit_h(checkpoint: str | Path | None = ...) -> Sam: ...
def build_sam_vit_l(checkpoint: str | Path | None = ...) -> Sam: ...
def build_sam_vit_b(checkpoint: str | Path | None = ...) -> Sam: ...

build_sam = build_sam_vit_h

sam_model_registry: dict[str, SamBuilder]
