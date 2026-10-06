"""Train/evaluate a fixed logistic baseline, then refit on every labelled site."""
import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, auc, brier_score_loss,
                             log_loss, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, FEATURE_VERSION, iter_sites


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_labels(path):
    labels = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"gene_id", "transcript_id", "transcript_position", "label"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Labels require columns {sorted(expected)}")
        for row in reader:
            key = (row["transcript_id"], row["transcript_position"])
            if key in labels:
                raise ValueError(f"Duplicate label key: {key}")
            if row["label"] not in {"0", "1"} or not row["gene_id"]:
                raise ValueError(f"Invalid binary label or empty gene ID for {key}")
            labels[key] = (row["gene_id"], int(row["label"]))
    return labels


def extract(data, labels_path, out):
    signature = {"data_sha256": digest(data), "labels_sha256": digest(labels_path),
                 "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES}
    cache, metadata_path = out / "features.npz", out / "features_metadata.json"
    if cache.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == signature:
        print("Using verified feature cache", flush=True)
        with np.load(cache, allow_pickle=False) as values:
            return {key: values[key] for key in values.files}, signature
    labels = read_labels(labels_path)
    features, targets, genes, transcripts, positions = [], [], [], [], []
    seen = set()
    for transcript, position, x in iter_sites(data):
        key = (transcript, position)
        if key in seen:
            raise ValueError(f"Duplicate signal site: {key}")
        seen.add(key)
        if key not in labels:
            raise ValueError(f"Signal site missing a label: {key}")
        gene, label = labels[key]
        features.append(x)
        targets.append(label)
        genes.append(gene)
        transcripts.append(transcript)
        positions.append(position)
        if len(features) % 20000 == 0:
            print(f"Extracted {len(features):,} sites", flush=True)
    missing = set(labels) - seen
    if missing:
        raise ValueError(f"{len(missing)} labelled sites missing from signals")
    if not features:
        raise ValueError("No sites found")
    values = {"X": np.asarray(features), "y": np.asarray(targets, dtype=np.int8),
              "genes": np.asarray(genes), "transcripts": np.asarray(transcripts),
              "positions": np.asarray(positions)}
    np.savez_compressed(cache, **values)
    metadata_path.write_text(json.dumps(signature, indent=2) + "\n")
    return values, signature


def model():
    # Fixed specification; the held-out test is not used for hyperparameter selection.
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))


def metrics(y, score):
    precision, recall, _ = precision_recall_curve(y, score)
    return {"roc_auc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
            "pr_auc_trapezoidal": float(auc(recall, precision)),
            "log_loss": float(log_loss(y, score)),
            "brier_score": float(brier_score_loss(y, score))}


def split_stats(y, genes, indices):
    return {"sites": len(indices), "genes": len(np.unique(genes[indices])),
            "positives": int(y[indices].sum()), "positive_fraction": float(y[indices].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    start = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    values, signature = extract(args.data, args.labels, out)
    X, y, genes = values["X"], values["y"], values["genes"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train, test = next(splitter.split(X, y, groups=genes))
    assert not set(genes[train]) & set(genes[test]), "Gene leakage"
    assert not set(values["transcripts"][train]) & set(values["transcripts"][test]), "Transcript leakage"
    for indices in (train, test):
        if len(np.unique(y[indices])) != 2:
            raise ValueError("Both classes must be present in each split")
    print(f"Training on {len(train):,} sites; testing on {len(test):,} sites ({X.shape[1]} features)", flush=True)
    heldout = model().fit(X[train], y[train])
    score = heldout.predict_proba(X[test])[:, 1]
    constant = np.full(len(test), y[train].mean())
    report = {"model": "StandardScaler + L2 logistic regression, C=1, no class weighting",
              "seed": args.seed, "split": "80/20 by gene; site fractions may differ",
              "n_features": X.shape[1], "train": split_stats(y, genes, train),
              "test": split_stats(y, genes, test), "heldout_metrics": metrics(y[test], score),
              "constant_baseline_metrics": metrics(y[test], constant),
              "gene_overlap": 0, "transcript_overlap": 0, "source": signature,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}}
    with (out / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["transcript_id", "transcript_position", "score", "label"])
        for i, p in zip(test, score):
            writer.writerow([values["transcripts"][i], values["positions"][i], float(p), int(y[i])])
    with (out / "split.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene_id", "transcript_id", "transcript_position", "label", "split"])
        test_set = set(test.tolist())
        for i in range(len(y)):
            writer.writerow([genes[i], values["transcripts"][i], values["positions"][i],
                             int(y[i]), "test" if i in test_set else "train"])
    bundle = {"pipeline": heldout, "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
              "sklearn_version": sklearn.__version__, "training_source": signature,
              "training_scope": "80% of data0 genes; held-out evaluation model"}
    joblib.dump(bundle, out / "evaluation_model.joblib")
    print(json.dumps(report["heldout_metrics"], indent=2), flush=True)
    print("Refitting unchanged model on all labelled sites for leaderboard use", flush=True)
    final = model().fit(X, y)
    bundle = dict(bundle, pipeline=final, training_scope="all data0 sites; submission model")
    joblib.dump(bundle, out / "baseline_model.joblib")
    with (out / "coefficients.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "coefficient_per_standard_deviation"])
        writer.writerows(zip(FEATURE_NAMES, final[-1].coef_[0].tolist()))
    report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved model and evaluation to {out.resolve()} in {report['elapsed_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
