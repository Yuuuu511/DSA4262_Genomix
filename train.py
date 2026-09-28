"""Train and evaluate a first-pass m6A site classifier."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import load_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="m6Anet data.json or data.json.gz")
    parser.add_argument("--labels", required=True, help="data.info with gene_id and label")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--model", choices=("logistic", "hgb"), default="hgb")
    args = parser.parse_args()

    sites = load_features(args.json)
    labels = pd.read_csv(args.labels)
    required = {"gene_id", "transcript_id", "transcript_position", "label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing label columns: {sorted(required - set(labels.columns))}")
    if labels.duplicated(["transcript_id", "transcript_position"]).any():
        raise ValueError("Duplicate transcript positions in labels")
    data = sites.merge(labels[list(required)], on=["transcript_id", "transcript_position"],
                       how="inner", validate="one_to_one")
    print(f"JSON sites: {len(sites)}; matched labelled sites: {len(data)}; "
          f"unmatched sites: {len(sites) - len(data)}")
    if len(data) < 20 or data.label.nunique() != 2:
        raise ValueError("Need at least 20 matched sites and both label classes")
    if not set(data.label.unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")

    columns = [c for c in sites if c not in ("transcript_id", "transcript_position")]
    categorical = ["motif", "left_flank", "right_flank"]
    numeric = [c for c in columns if c not in categorical]
    transformer = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("sequence", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    classifier = (LogisticRegression(max_iter=1000, class_weight="balanced")
                  if args.model == "logistic" else
                  HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                 l2_regularization=1, random_state=42))
    pipeline = make_pipeline(transformer, classifier)
    train_idx, test_idx = next(GroupShuffleSplit(test_size=.2, random_state=42).split(
        data, data.label, groups=data.gene_id))
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    if train.label.nunique() != 2 or test.label.nunique() != 2:
        raise ValueError("This group split lacks a positive or negative class; choose more data")
    pipeline.fit(train[columns], train.label)
    scores = pipeline.predict_proba(test[columns])[:, 1]
    metrics = {"json_sites": len(sites), "matched_sites": len(data),
               "train_sites": len(train), "test_sites": len(test),
               "train_genes": train.gene_id.nunique(), "test_genes": test.gene_id.nunique(),
               "train_positive_fraction": float(train.label.mean()),
               "test_positive_fraction": float(test.label.mean()),
               "roc_auc": float(roc_auc_score(test.label, scores)),
               "pr_auc": float(average_precision_score(test.label, scores)),
               "random_pr_auc": float(test.label.mean()), "model": args.model}
    # The saved model is fitted on the training genes only, keeping the test set independent.
    joblib.dump({"pipeline": pipeline, "columns": columns}, args.model_out)
    with open(args.metrics_out, "w") as output:
        json.dump(metrics, output, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
