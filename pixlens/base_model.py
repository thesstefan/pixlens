import abc
import pathlib

import yaml

from pixlens.utils.utils import hash_primitive_list
from pixlens.utils.yaml_constructible import YamlConstructible


class BaseModel(abc.ABC, YamlConstructible):
    @classmethod
    def get_model_name(cls) -> str:
        return cls.__name__

    @property
    @abc.abstractmethod
    def params_dict(self) -> dict[str, str | bool | int | float]:
        ...

    @property
    def params_hash(self) -> str:
        return hash_primitive_list(list(self.params_dict.keys()))

    @property
    def model_id(self) -> str:
        return self.get_model_name() + "_" + self.params_hash

    def to_yaml(self, yaml_path: pathlib.Path) -> None:
        data = {"class": self.get_model_name(), "params": self.params_dict}

        with yaml_path.open("w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
