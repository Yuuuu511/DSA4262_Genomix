from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "merged" / "merged_site_features.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"

RANDOM_STATE = 42

# These fractions apply approximately to the number of genes,
# not necessarily to the number of rows.
TEST_FRACTION = 0.20
VALIDATION_FRACTION_OF_REMAINING = 0.25

FEATURE_COLS = (
    ["read_count"]
    + [f"mean_{i}" for i in range(1, 10)]
    + [f"std_{i}" for i in range(1, 10)]
    + [f"median_{i}" for i in range(1, 10)]
)

# Select the Random Forest configuration using validation PR-AUC.
RF_CONFIGS = {
    "rf_leaf1": {
        "n_estimators": 300,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    "rf_leaf5": {
        "n_estimators": 300,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
}


# ============================================================
# 2. Load and validate data
# ============================================================

def load_data():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    data = pd.read_csv(
    INPUT_PATH,
    dtype={"gene_id": "string"},
    low_memory=False,
    )

    required = {
        "source_dataset",
        "sample_id",
        "gene_id",
        "transcript_id",
        "transcript_position",
        "label",
        *FEATURE_COLS,
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if data["sample_id"].isna().any():
        raise ValueError("Missing sample identifiers found")

    if data["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample identifiers found")

    data["label"] = pd.to_numeric(data["label"], errors="raise")

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    data["label"] = data["label"].astype(int)

    data["gene_id"] = (
        data["gene_id"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # Use only data0/data1 rows with known genes for this evaluation.
    # Do not replace missing gene IDs with transcript IDs.
    eligible = (
        data["source_dataset"].isin(["data0", "data1"])
        & data["gene_id"].notna()
    )

    excluded = data.loc[~eligible].copy()
    data = data.loc[eligible].copy().reset_index(drop=True)

    if data.empty:
        raise ValueError("No eligible rows remain")

    if data[["transcript_id", "transcript_position"]].isna().any().any():
        raise ValueError("Missing site coordinates found")

    # A transcript must map consistently to one gene across datasets.
    genes_per_transcript = data.groupby("transcript_id")["gene_id"].nunique()

    if (genes_per_transcript > 1).any():
        raise ValueError("Some transcripts map to multiple genes")

    for column in FEATURE_COLS:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data[FEATURE_COLS] = data[FEATURE_COLS].replace(
        [np.inf, -np.inf], np.nan
    )

    print(f"Eligible rows: {len(data):,}")
    print(f"Unique genes: {data['gene_id'].nunique():,}")
    print(f"Excluded rows: {len(excluded):,}")

    if not excluded.empty:
        print("\nExcluded rows by dataset:")
        print(excluded.groupby("source_dataset").size().to_string())

    return data, excluded


# ============================================================
# 3. Split by gene
# ============================================================

def split_by_gene(data):
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
    )

    development_idx, test_idx = next(
        outer_split.split(data, groups=data["gene_id"])
    )

    development = data.iloc[development_idx].copy()
    test = data.iloc[test_idx].copy()

    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION_OF_REMAINING,
        random_state=RANDOM_STATE + 1,
    )

    train_idx, validation_idx = next(
        inner_split.split(
            development,
            groups=development["gene_id"],
        )
    )

    train = development.iloc[train_idx].copy()
    validation = development.iloc[validation_idx].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    names = list(splits)

    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left = splits[left_name]
            right = splits[right_name]

            shared_genes = set(left["gene_id"]) & set(right["gene_id"])
            if shared_genes:
                raise ValueError("Gene overlap detected between splits")

            left_sites = set(
                zip(left["transcript_id"], left["transcript_position"])
            )
            right_sites = set(
                zip(right["transcript_id"], right["transcript_position"])
            )

            if left_sites & right_sites:
                raise ValueError("Site overlap detected between splits")

    for name, frame in splits.items():
        if frame["label"].nunique() != 2:
            raise ValueError(f"{name} does not contain both label classes")

        print(
            f"{name}: {len(frame):,} rows, "
            f"{frame['gene_id'].nunique():,} genes, "
            f"positive rate = {frame['label'].mean():.4f}"
        )

    return splits


# ============================================================
# 4. Define models
# ============================================================

def make_baseline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def make_random_forest(config):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                **config,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


# ============================================================
# 5. Evaluate probability scores
# ============================================================

def calculate_metrics(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)

    return {
        "n_sites": int(len(labels)),
        "positive_rate": float(np.mean(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        # Trapezoidal area under the precision-recall curve.
        "pr_auc": float(auc(recall, precision)),
        # AP is reported separately because it differs from trapezoidal AUC.
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
    }


def save_curves(labels, model_scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, scores in model_scores.items():
        metrics = calculate_metrics(labels, scores)

        fpr, tpr, _ = roc_curve(labels, scores)
        axes[0].plot(
            fpr,
            tpr,
            label=f"{name}: {metrics['roc_auc']:.3f}",
        )

        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[1].plot(
            recall,
            precision,
            label=f"{name}: {metrics['pr_auc']:.3f}",
        )

    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Held-out test ROC curves",
    )

    axes[1].axhline(
        np.mean(labels),
        linestyle="--",
        color="gray",
        label="Positive prevalence",
    )
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Held-out test precision-recall curves",
    )

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_curves.png", dpi=200)
    plt.close(fig)


# ============================================================
# 6. Train, select, evaluate, and save
# ============================================================

def main():
    data, excluded = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded[
        [
            "source_dataset",
            "sample_id",
            "transcript_id",
            "transcript_position",
            "gene_id",
        ]
    ].to_csv(OUTPUT_DIR / "excluded_rows.csv", index=False)

    print("\nCreating gene-disjoint splits...")
    splits = split_by_gene(data)

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    # Save the split assignment so all experiments can reuse it.
    manifest = pd.concat(
        [
            frame[
                [
                    "sample_id",
                    "source_dataset",
                    "gene_id",
                    "transcript_id",
                    "transcript_position",
                    "label",
                ]
            ].assign(split=name)
            for name, frame in splits.items()
        ],
        ignore_index=True,
    )
    manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)

    if train[FEATURE_COLS].isna().all().any():
        raise ValueError("A training feature contains only missing values")

    X_train = train[FEATURE_COLS]
    y_train = train["label"]

    X_validation = validation[FEATURE_COLS]
    y_validation = validation["label"]

    validation_results = []

    print("\nTraining Logistic Regression baseline...")
    baseline = make_baseline()
    baseline.fit(X_train, y_train)

    baseline_scores = baseline.predict_proba(X_validation)[:, 1]
    validation_results.append({
        "model": "logistic_regression",
        **calculate_metrics(y_validation, baseline_scores),
    })

    rf_results = []

    for name, config in RF_CONFIGS.items():
        print(f"\nTraining {name}...", flush=True)

        model = make_random_forest(config)
        model.fit(X_train, y_train)

        scores = model.predict_proba(X_validation)[:, 1]
        result = {
            "model": name,
            **calculate_metrics(y_validation, scores),
        }

        validation_results.append(result)
        rf_results.append(result)

        print(
            f"Validation ROC-AUC: {result['roc_auc']:.4f}; "
            f"PR-AUC: {result['pr_auc']:.4f}; "
            f"AP: {result['average_precision']:.4f}"
        )

    pd.DataFrame(validation_results).to_csv(
        OUTPUT_DIR / "validation_metrics.csv",
        index=False,
    )

    # Select the RF configuration without consulting the test set.
    best_rf_name = max(rf_results, key=lambda row: row["pr_auc"])["model"]
    best_config = RF_CONFIGS[best_rf_name]

    print(f"\nSelected Random Forest: {best_rf_name}")

    # Refit on train + validation before the final held-out evaluation.
    development = pd.concat([train, validation], ignore_index=True)

    evaluated_models = {
        "logistic_regression": make_baseline(),
        "random_forest": make_random_forest(best_config),
    }

    test_results = []
    test_scores = {}

    predictions = test[
        [
            "sample_id",
            "source_dataset",
            "gene_id",
            "transcript_id",
            "transcript_position",
            "label",
        ]
    ].copy()

    for name, model in evaluated_models.items():
        print(f"\nRefitting {name} on train + validation...", flush=True)

        model.fit(
            development[FEATURE_COLS],
            development["label"],
        )

        scores = model.predict_proba(test[FEATURE_COLS])[:, 1]
        test_scores[name] = scores
        predictions[f"{name}_score"] = scores

        test_results.append({
            "model": name,
            "dataset": "all_eligible_test",
            **calculate_metrics(test["label"], scores),
        })

        # Also report each test source separately.
        for source in sorted(test["source_dataset"].unique()):
            mask = test["source_dataset"].eq(source).to_numpy()
            source_labels = test.loc[mask, "label"]

            if source_labels.nunique() == 2:
                test_results.append({
                    "model": name,
                    "dataset": source,
                    **calculate_metrics(source_labels, scores[mask]),
                })

        # These saved models have never trained on the test set.
        joblib.dump(
            {
                "pipeline": model,
                "feature_cols": FEATURE_COLS,
                "training_scope": "train_and_validation_only",
                "random_state": RANDOM_STATE,
            },
            OUTPUT_DIR / f"{name}_evaluated.joblib",
        )

    metrics = pd.DataFrame(test_results)
    metrics.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    predictions.to_csv(
        OUTPUT_DIR / "heldout_test_predictions.csv",
        index=False,
    )

    save_curves(test["label"], test_scores)

    rf = evaluated_models["random_forest"].named_steps["model"]

    pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values(
        "importance", ascending=False
    ).to_csv(
        OUTPUT_DIR / "rf_feature_importance.csv",
        index=False,
    )

    config = {
        "input_path": str(INPUT_PATH),
        "random_state": RANDOM_STATE,
        "test_gene_fraction": TEST_FRACTION,
        "validation_gene_fraction_of_remaining": (
            VALIDATION_FRACTION_OF_REMAINING
        ),
        "selected_rf": best_rf_name,
        "rf_parameters": best_config,
        "feature_cols": FEATURE_COLS,
        "included_data": "data0 and data1 rows with known gene_id",
        "excluded_rows": int(len(excluded)),
        "pr_auc_definition": "Trapezoidal area under the PR curve",
        "m6anet_benchmark_included": False,
    }

    with (OUTPUT_DIR / "training_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)

    print("\nHeld-out test results:")
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(
        "\nThe test set was excluded from model fitting. "
        "Do not repeatedly tune models using these test results."
    )


if __name__ == "__main__":
    main()