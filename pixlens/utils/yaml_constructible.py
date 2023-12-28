import pathlib
from typing import Self

import yaml


class YamlConstructMismatchError(RuntimeError):
    def __init__(self, yaml_path: pathlib.Path | str, cls_name: str) -> None:
        super().__init__(f"{yaml_path} is not a valid config for {cls_name}!")


class YamlConstructible:
    """Constructs class from a JSON file.

    The JSON files should have the following structure:

    {
        'class': str,
        'params': [ ... ]
    }

    Esentially, type[class].__init__(**params) is returned.
    """

    @classmethod
    def from_yaml(cls, yaml_path: pathlib.Path | str) -> Self:
        if isinstance(yaml_path, str):
            yaml_path = pathlib.Path(yaml_path)

        with yaml_path.open("r") as file:
            params = yaml.safe_load(file)

        if params.get("class") != cls.__name__:
            raise YamlConstructMismatchError(yaml_path, cls.__name__)

        return cls(**params.get("params", {}))
