import itertools
import json
from pathlib import Path

import pandas as pd

from pixlens.dataset.edit_dataset import EditDataset, EditSchema
from pixlens.dataset.prompt_utils import (
    generate_description_based_prompt,
    generate_instruction_based_prompt,
)
from pixlens.evaluation.interfaces import (
    EditType,
)


class EditValDataset(EditDataset):
    json_object_path: Path
    dataset_path: Path

    def _get_image_path(self, category: str, image_id: str) -> Path:
        zero_prefixed_id = "0" * (12 - len(image_id)) + image_id

        return self.dataset_path / category / (zero_prefixed_id + ".jpg")

    # TODO: This is ugly :(
    def _create_edit_record(  # noqa: PLR0913
        self,
        edit_id: int,
        image_id: str,
        edit_type: EditType,
        category: str,
        from_attribute: str | None,
        to_attribute: str | None,
    ) -> dict[str, str | int | None]:
        return {
            "edit_id": edit_id,
            "image_id": image_id,
            "edit_type": edit_type,
            "category": category,
            "from_attribute": from_attribute,
            "to_attribute": to_attribute,
            "image_path": str(
                self._get_image_path(category, image_id),
            ),
            "instruction_prompt": generate_instruction_based_prompt(
                edit_type,
                from_attribute,
                to_attribute,
                category,
            ),
            "description_prompt": generate_description_based_prompt(
                edit_type,
                from_attribute,
                to_attribute,
                category,
            ),
        }

    def _init_df(self, edits_path: Path) -> None:
        # We have to create the Edits CSV from a EditVal-like object.json file.
        # See an example here of such a file here
        #   https://github.com/deep-ml-research/editval_code/blob/main/object.json

        with self.json_object_path.open() as json_file:
            json_data = json.load(json_file)

        edit_records: list[dict[str, str | int | None]] = []

        for category, images in json_data.items():
            for image_id, edits in images.items():
                # Record for removing the main object from the image
                edit_records.append(
                    self._create_edit_record(
                        len(edit_records),
                        image_id,
                        EditType.OBJECT_REMOVAL,
                        category,
                        None,
                        None,
                    ),
                )

                for edit_type, values in edits.items():
                    from_attributes = values.get("from", [""])
                    to_attributes = values.get("to", [])

                    for to_attribute, from_attribute in itertools.product(
                        to_attributes,
                        from_attributes,
                    ):
                        edit_records.append(
                            self._create_edit_record(
                                len(edit_records),
                                image_id,
                                edit_type,
                                category,
                                from_attribute,
                                to_attribute,
                            ),
                        )

        # TODO: Fix weird typing error
        self.edits_df = EditSchema.validate(pd.DataFrame(edit_records))  # type: ignore[assignment]
        self.edits_df.to_csv(edits_path)

    @property
    def name(self) -> str:
        return "EditVal"

    def __init__(
        self,
        json_object_path: Path,
        dataset_path: Path,
        edits_path: Path | None = None,
    ) -> None:
        self.json_object_path = json_object_path
        self.dataset_path = dataset_path

        super().__init__(edits_path)
