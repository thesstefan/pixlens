import os
import pathlib
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv

basedir = pathlib.Path(__file__).parent
load_dotenv(basedir / ".env")


# TODO: Allow more specific/granular model parameters.
@dataclass
class Config:
    DETECT_SEGMENT_MODEL: str | None = field(
        default=os.getenv("DETECT_SEGMENT_MODEL"),
    )
    DETECT_SEGMENT_CONFIDENCE_THRESHOLD: str | None = field(
        default=os.getenv("DETECT_CONFIDENCE_THRESHOLD"),
    )
    EDIT_MODEL: str | None = field(default=os.getenv("EDIT_MODEL"))
    DEVICE: torch.device = field(
        default=torch.device(
            os.getenv("DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu"),
        ),
    )
