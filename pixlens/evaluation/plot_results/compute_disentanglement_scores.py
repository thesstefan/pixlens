import json
from pathlib import Path

results: dict = {}


def compute_all_scores(
    data: dict,
    save_path: Path | None = None,
) -> None:
    model_name = next(iter(data.keys()))
    results[model_name] = {}
    data = data[model_name]
    attributes = ["texture", "color", "style", "pattern"]
    results[model_name]["Avg_norm"] = data["Avg_norm"]
    results[model_name]["Avg_norm_per_attribute"] = {}
    results[model_name]["Inter_sample_and_intra_attribute"] = {}

    for attribute in attributes:
        results[model_name]["Inter_sample_and_intra_attribute"][attribute] = {}
        results[model_name]["Avg_norm_per_attribute"][attribute] = data[
            "Avg_norm_per_attribute"
        ][attribute]
        results[model_name]["Inter_sample_and_intra_attribute"][attribute][
            "Average Cosine Similarity"
        ] = data["Inter_sample_and_intra_attribute"][attribute][
            "Average Cosine Similarity"
        ]
        results[model_name]["Inter_sample_and_intra_attribute"][attribute][
            "Average Angle"
        ] = data["Inter_sample_and_intra_attribute"][attribute]["Average Angle"]

    results[model_name]["Inter_attribute"] = data["Inter_attribute"]
    if save_path is not None:
        with Path.open(
            save_path,
        ) as f:
            json.dump(results, f, indent=4)
    json.dump(
        results,
        Path.open(Path("results_disentanglement.json")),
        indent=4,
    )
