from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "benchmark_results"
KEYS = ["source_dataset", "transcript_id", "transcript_position"]


def calculate_metrics(labels, scores):
    if labels.nunique() != 2:
        raise ValueError("Both label classes are required")

    precision, recall, _ = precision_recall_curve(labels, scores)

    return {
        "n_sites": len(labels),
        "positive_rate": labels.mean(),
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": auc(recall, precision),
        "average_precision": average_precision_score(labels, scores),
    }


def main():
    heldout = pd.read_csv(
        BASE_DIR / "model_outputs" / "heldout_test_predictions.csv",
        dtype={"transcript_id": str, "gene_id": str},
    )

    if heldout.duplicated(KEYS).any():
        raise ValueError("Duplicate held-out observations found")

    parts = []

    for source in ["data0", "data1"]:
        path = (
            BASE_DIR / "m6anet_outputs" / source / "data.site_proba.csv"
        )
        predictions = pd.read_csv(
            path, dtype={"transcript_id": str}
        )

        predictions["source_dataset"] = source
        predictions = predictions.rename(
            columns={"probability_modified": "m6anet_score"}
        )
        parts.append(predictions[KEYS + ["m6anet_score"]])

    m6anet = pd.concat(parts, ignore_index=True)

    if m6anet.duplicated(KEYS).any():
        raise ValueError("Duplicate m6Anet predictions found")

    # Match each dataset separately using the source identifier.
    matched = heldout.merge(
        m6anet,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unexpected = matched["_merge"].eq("right_only")
    if unexpected.any():
        raise ValueError("m6Anet returned sites outside the held-out set")

    missing = matched.loc[
        matched["_merge"].eq("left_only"),
        KEYS + ["label"],
    ].copy()

    common = matched.loc[
        matched["_merge"].eq("both")
    ].drop(columns="_merge").copy()

    if common.empty:
        raise ValueError("No common test observations found")

    model_columns = {
        "logistic_regression": "logistic_regression_score",
        "random_forest": "random_forest_score",
        "m6anet_pretrained": "m6anet_score",
    }

    for column in model_columns.values():
        values = pd.to_numeric(common[column], errors="raise")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f"Invalid probability scores: {column}")
        common[column] = values

    subsets = {"all_common_test": common}
    for source, frame in common.groupby("source_dataset"):
        subsets[source] = frame

    results = []

    for dataset, frame in subsets.items():
        for model, column in model_columns.items():
            results.append({
                "model": model,
                "dataset": dataset,
                **calculate_metrics(frame["label"], frame[column]),
            })

    metrics = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        OUTPUT_DIR / "comparison_metrics.csv", index=False
    )
    common.to_csv(
        OUTPUT_DIR / "common_test_predictions.csv", index=False
    )
    missing.to_csv(
        OUTPUT_DIR / "missing_m6anet_sites.csv", index=False
    )

    print(f"Original held-out observations: {len(heldout):,}")
    print(f"Common observations: {len(common):,}")
    print(f"Missing m6Anet predictions: {len(missing):,}")
    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(
        "\nThis evaluates the official pretrained m6Anet model. "
        "Overlap with its original training data has not been verified."
    )


if __name__ == "__main__":
    main()