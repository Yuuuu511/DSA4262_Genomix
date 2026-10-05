"""One final evaluation of validation-selected method and fixed controls."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from modeling import evaluation_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    output = root / "test_metrics.csv"
    if output.exists():
        raise SystemExit("Test was already evaluated. Inspect existing test_metrics.csv.")
    summary = json.loads((root / "experiment.json").read_text())
    data = joblib.load(root / summary["cache_file"])
    test = data.loc[data.split.eq("test")]
    rows = []
    names = list(dict.fromkeys(["logistic_simple", "hgb_original", summary["selected_model"]]))
    for name in names:
        bundle = joblib.load(root / f"{name}.joblib")
        scores = bundle["pipeline"].predict_proba(test[bundle["columns"]])[:, 1]
        rows.extend(evaluation_rows(test, scores, name))
        predictions = test[["source_id", "transcript_id", "transcript_position", "label"]].copy()
        predictions["score"] = scores
        predictions.to_csv(root / f"test_predictions_{name}.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    summary["test_evaluated"] = True
    (root / "experiment.json").write_text(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
