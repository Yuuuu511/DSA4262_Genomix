"""Compare validation scores at different actual JSON read coverages."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from modeling import MODELS, metrics_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    summary = json.loads((root / "experiment.json").read_text())
    data = joblib.load(root / summary["cache_file"])
    val = data.loc[data.split.eq("validation")]
    bands = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, np.inf)]
    rows = []
    for name in MODELS:
        bundle = joblib.load(root / f"{name}.joblib")
        scores = bundle["pipeline"].predict_proba(val[bundle["columns"]])[:, 1]
        for dataset in ["combined", *sorted(val.source_id.unique())]:
            for low, high in bands:
                mask = val.read_count.ge(low).to_numpy() & val.read_count.lt(high).to_numpy()
                if dataset != "combined":
                    mask &= val.source_id.to_numpy() == dataset
                rows.append({"model": name, "dataset": str(dataset),
                             "coverage": f"{low}-{int(high)-1}" if np.isfinite(high) else f"{low}+",
                             **metrics_for(val.loc[mask], scores[mask])})
    table = pd.DataFrame(rows)
    table.to_csv(root / "validation_by_coverage.csv", index=False)
    print("Wrote validation_by_coverage.csv. These are descriptive strata, not new test sets.")


if __name__ == "__main__":
    main()
