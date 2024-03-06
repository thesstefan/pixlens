import logging
from pathlib import Path

from pixlens.dataset.edit_dataset import EditDataset
from pixlens.editing.interfaces import (
    ImageEditingPromptType,
    PromptableImageEditingModel,
)
from pixlens.evaluation.interfaces import Edit
from pixlens.evaluation.utils import prompt_to_filename
from pixlens.utils.utils import get_cache_dir


class PreprocessingPipeline:
    edit_dataset: EditDataset

    def __init__(self, edit_dataset: EditDataset) -> None:
        self.edit_dataset = edit_dataset

    def get_edited_image_path(
        self,
        input_image_path: Path,
        prompt: str,
        dataset_name: str,
        editing_model: PromptableImageEditingModel,
    ) -> Path:
        return (
            get_cache_dir()
            / editing_model.model_id
            / dataset_name
            / input_image_path.stem
            / prompt_to_filename(prompt)
        ).with_suffix(".png")

    def save_edited_image(
        self,
        edit: Edit,
        prompt: str,
        dataset_name: str,
        editing_model: PromptableImageEditingModel,
    ) -> None:
        edited_image_path = self.get_edited_image_path(
            Path(edit.image_path),
            prompt,
            dataset_name,
            editing_model,
        )

        if edited_image_path.exists():
            logging.info(
                "Image already exists at %s...",
                edited_image_path,
            )

            return

        logging.info("Editing image...")

        edited_image_path.parent.mkdir(parents=True, exist_ok=True)
        edited_image = editing_model.edit_image(prompt, edit.image_path, edit)
        edited_image.save(edited_image_path)

        logging.info("Image saved to %s", edited_image_path)

    def init_model_dir(self, model: PromptableImageEditingModel) -> None:
        model_dir = get_cache_dir() / model.model_id
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        model.to_yaml(model_dir / "model_params.yaml")

    def execute_pipeline(
        self,
        models: list[PromptableImageEditingModel],
    ) -> None:
        for model in models:
            logging.info("Running model: %s", model.get_model_name())

            self.init_model_dir(model)

            for edit in self.edit_dataset:
                prompt = (
                    edit.instruction_prompt
                    if model.prompt_type == ImageEditingPromptType.INSTRUCTION
                    else edit.description_prompt
                )

                logging.info("Prompt: %s", prompt)
                logging.info("image_path: %s", edit.image_path)

                self.save_edited_image(
                    edit,
                    prompt,
                    self.edit_dataset.name,
                    model,
                )

                logging.info("")
