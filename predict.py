"""Predict one score per transcript position and write the required CSV."""
import argparse

import joblib

from features import load_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sites = load_features(args.json)
    saved = joblib.load(args.model)  # Only load model files created by your own training run.
    scores = saved["pipeline"].predict_proba(sites[saved["columns"]])[:, 1]
    result = sites[["transcript_id", "transcript_position"]].copy()
    result["score"] = scores
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} site scores to {args.output}")


if __name__ == "__main__":
    main()
