import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pixlens.dataset.edit_dataset import EditDataset, EditSchema
from pixlens.dataset.prompt_utils import (
    generate_description_based_prompt,
    generate_instruction_based_prompt,
)
from pixlens.evaluation.interfaces import EditType


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
        raw_df = pd.DataFrame(edit_records)
        raw_df = raw_df.replace({np.nan: None})

        # reorder rows in raw_df so that all the object removal
        # edits are at the bottom
        object_removal_edits = raw_df[
            raw_df["edit_type"] == EditType.OBJECT_REMOVAL
        ]

        # object removal edits should be ordered alphabetically by category
        object_removal_edits = object_removal_edits.sort_values(
            by=["category", "image_id"],
        )

        non_object_removal_edits = raw_df[
            raw_df["edit_type"] != EditType.OBJECT_REMOVAL
        ]
        raw_df = pd.concat([non_object_removal_edits, object_removal_edits])

        # edit ids are not in order, so we reset them
        raw_df["edit_id"] = range(len(raw_df))

        self.edits_df = EditSchema.validate(raw_df)  # type: ignore[assignment]
        self.edits_df.to_csv(edits_path, index=False)

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
