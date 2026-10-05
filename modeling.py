import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import SIGNALS

CATEGORICAL = ["motif", "left_flank", "right_flank"]
MODELS = ["logistic_simple", "hgb_original", "hgb_normalized", "hgb_distribution"]


def model_columns(frame, name):
    original = ["read_count", *CATEGORICAL,
                *[f"{s}_{stat}" for s in SIGNALS
                  for stat in ("mean", "std", "q25", "median", "q75")]]
    if name == "logistic_simple":
        return [*[f"{s}_mean" for s in SIGNALS], *CATEGORICAL]
    if name == "hgb_original":
        return original
    normalized = [c for c in frame if c.startswith("norm_")] + ["reference_fallback"]
    if name == "hgb_normalized":
        return original + normalized
    return original + normalized + [c for c in frame if c.startswith("dist_")]


def build_model(columns, name):
    numeric = [c for c in columns if c not in CATEGORICAL]
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("sequence", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    classifier = (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
                  if name == "logistic_simple" else
                  HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                 l2_regularization=1, random_state=42,
                                                 early_stopping=False))
    return make_pipeline(transform, classifier)


def metrics_for(frame, scores):
    if frame.empty:
        return {"sites": 0}
    y = frame.label.to_numpy()
    out = {"sites": len(frame), "positives": int(np.sum(y)), "positive_fraction": float(np.mean(y)),
           "pr_metric": "average_precision",
           "roc_auc": None, "pr_auc": None}
    if len(np.unique(y)) == 2:
        out.update(roc_auc=float(roc_auc_score(y, scores)),
                   pr_auc=float(average_precision_score(y, scores)))
    return out


def evaluation_rows(frame, scores, name):
    rows = [{"model": name, "dataset": "combined", **metrics_for(frame, scores)}]
    for source in sorted(frame.source_id.unique()):
        mask = frame.source_id.to_numpy() == source
        rows.append({"model": name, "dataset": str(source),
                     **metrics_for(frame.loc[mask], scores[mask])})
    return rows
