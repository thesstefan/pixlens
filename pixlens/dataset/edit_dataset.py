import abc
import logging
from collections.abc import Iterator
from pathlib import Path

import dacite
import pandas as pd
import pandera as pa
from pandera.typing import Series

from pixlens.evaluation.interfaces import Edit, EditType
from pixlens.utils.utils import get_cache_dir


def _df_to_edits(df: pd.DataFrame) -> list[Edit]:
    # NOTE: iterrows is usually really slow. May need to
    # revisit this if we deal with a lot of elements
    return [
        dacite.from_dict(
            Edit,
            row.to_dict(),
            config=dacite.Config(
                cast=[EditType],
            ),
        )
        for _, row in df.iterrows()
    ]


class EditSchema(pa.DataFrameModel):
    edit_id: Series[int] = pa.Field(ge=0)
    image_id: Series[str]
    edit_type: Series[str] = pa.Field(
        isin=[edit_type.value for edit_type in EditType],
    )
    category: Series[str]
    from_attribute: Series[str] = pa.Field(nullable=True)
    to_attribute: Series[str] = pa.Field(nullable=True)
    image_path: Series[str]
    instruction_prompt: Series[str]
    description_prompt: Series[str]


class EditDataset(abc.ABC):
    edits_df: pa.typing.DataFrame[EditSchema]
    schema: pa.DataFrameSchema

    @abc.abstractmethod
    def _init_df(self, edits_path: Path) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def __init__(self, edits_path: Path | None) -> None:
        if edits_path and edits_path.exists():
            try:
                # TODO: Fix weird typing error
                self.edits_df = EditSchema.validate(  # type: ignore[assignment]
                    pd.read_csv(edits_path),
                    lazy=True,
                )
            except pa.errors.SchemaErrors as err:
                logging.warning("Schema errors and failure cases:")
                logging.warning(err)
                logging.warning(
                    "Deleting cached edit dataset, as it does not comply "
                    "with the established schema",
                )
                edits_path.unlink()
            else:
                return

        if not edits_path:
            edits_path = get_cache_dir() / (self.name + "_edits.csv")

        self._init_df(edits_path)

    def get_edit(self, edit_id: int) -> Edit:
        return _df_to_edits(self.edits_df[self.edits_df["edit_id"] == edit_id])[
            0
        ]

    def get_all_edits(self) -> list[Edit]:
        return _df_to_edits(self.edits_df)

    def get_all_edits_with_type(self, edit_type: EditType) -> list[Edit]:
        return _df_to_edits(
            self.edits_df[self.edits_df["edit_type"] == edit_type],
        )

    def __iter__(self) -> Iterator[Edit]:
        # NOTE: iterrows is usually really slow. May need to
        # revisit this if we deal with a lot of elements
        for _, row in self.edits_df.iterrows():
            yield dacite.from_dict(
                Edit,
                row.to_dict(),
                config=dacite.Config(
                    cast=[EditType],
                ),
            )
