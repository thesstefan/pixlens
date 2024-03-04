import dataclasses
import enum
import json
import pathlib

import ImageReward as RM  # noqa: N814

from pixlens.evaluation.interfaces import (
    EvaluationInput,
    EvaluationOutput,
    OperationEvaluation,
)


class RealismModelType(enum.Enum):
    BASE = "ImageReward-v1.0"


@dataclasses.dataclass(kw_only=True)
class RealismEvaluationOutput(EvaluationOutput):
    realism_score: float = 0.0

    def persist(self, save_dir: pathlib.Path) -> None:
        save_dir = save_dir / "realism_preservation"
        save_dir.mkdir(parents=True, exist_ok=True)

        score_summary = {
            "success": True,
            "realism_score": self.realism_score,
        }
        json_str = json.dumps(score_summary, indent=4)
        score_json_path = save_dir / "scores.json"

        with score_json_path.open("w") as score_json:
            score_json.write(json_str)


class RealismEvaluation(OperationEvaluation):
    def __init__(
        self,
        reward_model: RealismModelType = RealismModelType.BASE,
    ) -> None:
        self.reward_model = RM.load(reward_model)

    def evaluate_edit(
        self,
        evaluation_input: EvaluationInput,
    ) -> RealismEvaluationOutput:
        reward = self.reward_model.score(
            prompt=evaluation_input.edit.category,
            image=evaluation_input.edited_image,
        )
        return RealismEvaluationOutput(
            success=True,
            edit_specific_score=reward,
        )
