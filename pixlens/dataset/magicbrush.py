import json
from pathlib import Path

import pandas as pd

from pixlens.dataset.edit_dataset import EditDataset, EditSchema
from pixlens.dataset.prompt_utils import PROMPT_SEP


class MagicBrushDataset(EditDataset):
    def _get_descriptions(self) -> dict[str, str]:
        global_desc_path = self.magicbrush_dir / "global_descriptions.json"
        descriptions: dict[str, str] = {}

        # Only the 'dev' version contains global descriptions
        if global_desc_path.is_file():
            with global_desc_path.open() as json_file:
                global_descriptors = json.load(json_file)
                for edit_description in global_descriptors.values():
                    descriptions |= edit_description

        return descriptions

    def _init_df(self, edits_path: Path) -> None:
        descriptions = self._get_descriptions()

        with (self.magicbrush_dir / "edit_turns.json").open() as json_file:
            edit_turns = json.load(json_file)

        with (self.prompt_info_extension_json).open() as json_file:
            prompt_info = json.load(json_file)

        edit_records: list[dict[str, str | int | None]] = []
        edit_id = 0

        for edit in edit_turns:
            image = edit["input"]

            if image in prompt_info:
                info = prompt_info[edit["input"]]
                image_id = image.partition("-")[0]

                edit_records.append(
                    {
                        "edit_id": edit_id,
                        # Format of image paths is {COCO_ID}-{STUFF}.png
                        "image_id": image_id,
                        "edit_type": info["edit_type"],
                        "category": info["category"],
                        "from_attribute": info["from_attribute"],
                        "to_attribute": info["to_attribute"],
                        "image_path": str(
                            self.magicbrush_dir / "images" / image_id / image,
                        ),
                        "instruction_prompt": edit["instruction"],
                        # TODO: Find a better way to handle this. What will
                        # happen is that description-based models won't find
                        # a [SEP] and would crash.
                        #
                        # That's fine for now, but we should find a way to tell
                        # to the user that description based models can be
                        # benchmarked only by using the MagicBrush dev set.
                        "description_prompt": (
                            descriptions[image]
                            + PROMPT_SEP
                            + descriptions[edit["output"]]
                        )
                        if descriptions
                        else "",
                    },
                )

        # TODO: Fix weird typing error
        self.edits_df = EditSchema.validate(pd.DataFrame(edit_records))  # type: ignore[assignment]
        self.edits_df.to_csv(edits_path, index=False)

    @property
    def name(self) -> str:
        return "MagicBrush"

    def __init__(
        self,
        magicbrush_dir: Path,
        prompt_info_extension_json: Path,
        edits_path: Path | None = None,
    ) -> None:
        self.magicbrush_dir = magicbrush_dir
        self.prompt_info_extension_json = prompt_info_extension_json

        super().__init__(edits_path)
