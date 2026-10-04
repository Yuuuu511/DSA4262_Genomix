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


def load_label_tables(paths, gene_map_path=None):
    required = {"transcript_id", "transcript_position", "label"}
    tables = []
    mappings = []
    for path in paths:
        table = pd.read_csv(path)
        if not required.issubset(table.columns):
            raise ValueError(f"{path}: Missing label columns: {sorted(required - set(table.columns))}")
        tables.append(table)
        if "gene_id" in table:
            mappings.append(table[["transcript_id", "gene_id"]])
    if gene_map_path:
        mapping = pd.read_csv(gene_map_path)
        if not {"transcript_id", "gene_id"}.issubset(mapping.columns):
            raise ValueError("--gene-map CSV needs transcript_id and gene_id")
        mappings.append(mapping[["transcript_id", "gene_id"]])
    if mappings:
        mapping = pd.concat(mappings, ignore_index=True).dropna().drop_duplicates()
        mapping = mapping.loc[mapping.gene_id.astype(str).str.strip().ne("")]
        if mapping.groupby("transcript_id").gene_id.nunique().gt(1).any():
            raise ValueError("Inconsistent transcript-to-gene mapping")
        gene_lookup = mapping.set_index("transcript_id").gene_id
    else:
        gene_lookup = pd.Series(dtype=object)
    for table in tables:
        table["gene_id"] = table.transcript_id.map(gene_lookup)
    return tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, nargs="+",
                        help="One or more JSON paths, paired in order with --labels")
    parser.add_argument("--labels", required=True, nargs="+",
                        help="One label file per JSON; gene_id is optional for transcript split")
    parser.add_argument("--split-by", choices=("gene", "transcript"), default="gene",
                        help="gene requires complete mapping; transcript has weaker leakage protection")
    parser.add_argument("--gene-map", help="Optional CSV containing transcript_id,gene_id")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--model", choices=("logistic", "hgb"), default="hgb")
    args = parser.parse_args()

    if len(args.json) != len(args.labels):
        parser.error("--json and --labels must contain the same number of paths")
    required = {"gene_id", "transcript_id", "transcript_position", "label"}
    label_tables = load_label_tables(args.labels, args.gene_map)
    keys = ["transcript_id", "transcript_position"]
    frames = []
    dataset_stats = []
    columns = None
    for source_id, (json_path, label_path) in enumerate(zip(args.json, args.labels)):
        sites = load_features(json_path)
        current_columns = [c for c in sites if c not in keys]
        if columns is None:
            columns = current_columns
        elif columns != current_columns:
            raise ValueError(f"Inconsistent feature columns in {json_path}")
        labels = label_tables[source_id]
        if not required.issubset(labels.columns):
            raise ValueError(f"{label_path}: Missing label columns: "
                             f"{sorted(required - set(labels.columns))}")
        if labels[keys].isna().any().any() or labels.duplicated(keys).any():
            raise ValueError(f"{label_path}: Missing or duplicate transcript positions")
        frame = sites.merge(labels[sorted(required)], on=keys,
                            how="inner", validate="one_to_one")
        if frame.empty:
            raise ValueError(f"No labelled sites matched in {json_path}")
        if args.split_by == "gene" and frame.gene_id.isna().any():
            missing = frame.loc[frame.gene_id.isna(), "transcript_id"].nunique()
            raise ValueError(f"{label_path}: {missing} transcripts have no gene_id. "
                             "Provide a complete --gene-map CSV, or explicitly use "
                             "--split-by transcript (weaker leakage protection).")
        if frame.label.isna().any() or not set(frame.label.unique()).issubset({0, 1}):
            raise ValueError(f"{label_path}: Labels must be 0 or 1")
        frame["source_id"] = source_id  # Metadata only; never supplied to the classifier.
        frames.append(frame)
        stats = {"source_id": source_id, "json": json_path, "labels": label_path,
                 "json_sites": len(sites), "matched_sites": len(frame),
                 "unmatched_sites": len(sites) - len(frame)}
        dataset_stats.append(stats)
        print(json.dumps(stats), flush=True)
    data = pd.concat(frames, ignore_index=True)
    # A transcript must have one consistent gene mapping across all sources.
    if data.groupby("transcript_id").gene_id.nunique().gt(1).any():
        raise ValueError("Inconsistent gene_id mapping across label files for a transcript")
    overlapping = int(data.duplicated(keys).sum())
    print(f"Combined observations: {len(data)}; unique sites: "
          f"{len(data.drop_duplicates(keys))}; repeated-site observations: {overlapping}",
          flush=True)
    if len(data) < 20 or data.label.nunique() != 2:
        raise ValueError("Need at least 20 matched sites and both label classes")
    if not set(data.label.unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")

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
    group_column = "gene_id" if args.split_by == "gene" else "transcript_id"
    if args.split_by == "transcript":
        print("Using transcript-level split: different transcripts of the same gene "
              "may occur in both training and test sets.", flush=True)
    train_idx, test_idx = next(GroupShuffleSplit(test_size=.2, random_state=42).split(
        data, data.label, groups=data[group_column]))
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    if train.label.nunique() != 2 or test.label.nunique() != 2:
        raise ValueError("This group split lacks a positive or negative class; choose more data")
    pipeline.fit(train[columns], train.label)
    scores = pipeline.predict_proba(test[columns])[:, 1]
    metrics = {"datasets": dataset_stats,
               "json_sites": sum(s["json_sites"] for s in dataset_stats),
               "matched_sites": len(data), "unique_sites": len(data.drop_duplicates(keys)),
               "repeated_site_observations": overlapping,
               "train_sites": len(train), "test_sites": len(test),
               "split_by": args.split_by,
               "train_groups": train[group_column].nunique(),
               "test_groups": test[group_column].nunique(),
               "unmapped_gene_sites": int(data.gene_id.isna().sum()),
               "train_genes": train.gene_id.nunique(), "test_genes": test.gene_id.nunique(),
               "train_positive_fraction": float(train.label.mean()),
               "test_positive_fraction": float(test.label.mean()),
               "roc_auc": float(roc_auc_score(test.label, scores)),
               "pr_auc": float(average_precision_score(test.label, scores)),
               "random_pr_auc": float(test.label.mean()), "model": args.model}
    metrics["per_dataset_test"] = []
    for source_id in range(len(frames)):
        mask = test.source_id.to_numpy() == source_id
        subset = test.loc[mask]
        entry = {"source_id": source_id, "test_sites": len(subset)}
        if len(subset):
            entry["positive_fraction"] = float(subset.label.mean())
        if subset.label.nunique() == 2:
            entry["roc_auc"] = float(roc_auc_score(subset.label, scores[mask]))
            entry["pr_auc"] = float(average_precision_score(subset.label, scores[mask]))
        metrics["per_dataset_test"].append(entry)
    # The saved model is fitted only on the training partition.
    joblib.dump({"pipeline": pipeline, "columns": columns}, args.model_out)
    with open(args.metrics_out, "w") as output:
        json.dump(metrics, output, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
