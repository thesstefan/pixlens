import os
import pathlib
from dataclasses import dataclass, field

from dotenv import load_dotenv

basedir = pathlib.Path(__file__).parent
load_dotenv(basedir / ".env")


@dataclass
class Config:
    INFERENCE_TYPE: str = field(default=os.environ["INFERENCE_TYPE"])
    MODEL_PARAMS_YAML: str = field(default=os.environ["MODEL_PARAMS_YAML"])
