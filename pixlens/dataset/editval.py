import itertools
import json
from pathlib import Path

import pandas as pd

from pixlens.dataset.edit_dataset import EditDataset, EditSchema
from pixlens.evaluation.interfaces import (
    EditType,
)


class EditValDataset(EditDataset):
    json_object_path: Path
    dataset_path: Path

    def _get_image_path(self, obj_class: str, image_id: str) -> Path:
        zero_prefixed_id = "0" * (12 - len(image_id)) + image_id

        return self.dataset_path / obj_class / (zero_prefixed_id + ".jpg")

    def _init_df(self, edits_path: Path) -> None:
        # We have to create the Edits CSV from a EditVal-like object.json file.
        # See an example here of such a file here
        #   https://github.com/deep-ml-research/editval_code/blob/main/object.json

        with self.json_object_path.open() as json_file:
            json_data = json.load(json_file)

        edit_records: list[dict[str, str | int | None]] = []

        for obj_class, images in json_data.items():
            for image_id, edits in images.items():
                # Record for removing the main object from the image
                edit_records.append(
                    {
                        "edit_id": len(edit_records),
                        "image_id": image_id,
                        "edit_type": EditType.OBJECT_REMOVAL,
                        "category": obj_class,
                        "from_attribute": None,
                        "to_attribute": None,
                        "image_path": str(
                            self._get_image_path(obj_class, image_id),
                        ),
                    },
                )

                for edit_type, values in edits.items():
                    from_values = values.get("from", [""])
                    to_values = values.get("to", [])

                    for to_val, from_val in itertools.product(
                        to_values,
                        from_values,
                    ):
                        edit_records.append(
                            {
                                "edit_id": len(edit_records),
                                "image_id": image_id,
                                "edit_type": edit_type,
                                "category": obj_class,
                                "from_attribute": from_val,
                                "to_attribute": to_val,
                                "image_path": str(
                                    self._get_image_path(obj_class, image_id),
                                ),
                            },
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
