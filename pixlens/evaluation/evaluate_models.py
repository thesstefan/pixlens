import logging
import json
from pathlib import Path


from PIL import Image
import pandas as pd

from pixlens.editing.interfaces import PromptableImageEditingModel
from pixlens.evaluation import interfaces
from pixlens.evaluation.edit_dataset import PreprocessingPipeline
from pixlens.utils.utils import get_cache_dir, get_image_extension


class EvaluationPipeline:
    def __init__(self) -> None:
        self.edit_dataset: pd.DataFrame
        self.get_edit_dataset()

    def get_edit_dataset(self) -> None:
        pandas_path = Path(get_cache_dir(), "edit_dataset.csv")
        if pandas_path.exists():
            self.edit_dataset = pd.read_csv(pandas_path)
        else:
            raise FileNotFoundError

    def get_input_image_from_edit_id(self, edit_id: int) -> Image.Image:
        image_path = self.edit_dataset.iloc[edit_id]["input_image_path"]
        image_path = Path(image_path)
        image_extension = get_image_extension(image_path)
        return Image.open(image_path.with_suffix(image_extension))

    def get_edited_image_from_edit(
        self, edit: interfaces.Edit, model: PromptableImageEditingModel
    ) -> Image.Image:
        prompt = PreprocessingPipeline.generate_prompt(edit)
        edit_path = Path(
            get_cache_dir(),
            "models--" + model.get_model_name(),
            f"000000{str(edit.image_id)}",
            prompt,
        )
        extension = get_image_extension(edit_path)
        return Image.open(edit_path.with_suffix(extension))
