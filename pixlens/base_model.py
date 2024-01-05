from pixlens.utils.yaml_constructible import YamlConstructible


class BaseModel(YamlConstructible):
    @classmethod
    def get_model_name(cls) -> str:
        return cls.__name__
