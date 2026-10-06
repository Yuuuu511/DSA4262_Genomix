"""Write leaderboard-format probabilities for every site in new JSON-lines data."""
import argparse
import csv
import os
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn

from features import FEATURE_NAMES, FEATURE_VERSION, iter_sites


def predict(data, model_path, output, batch_size=2048):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    bundle = joblib.load(model_path)
    if bundle["feature_version"] != FEATURE_VERSION or bundle["feature_names"] != FEATURE_NAMES:
        raise ValueError("Model and feature extractor versions differ")
    if bundle["sklearn_version"] != sklearn.__version__:
        warnings.warn("Model was trained with a different scikit-learn version; use requirements.txt")
    pipeline = bundle["pipeline"]
    output = Path(output)
    if output.resolve() in {Path(data).resolve(), Path(model_path).resolve()}:
        raise ValueError("Output must not overwrite input data or the model")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, keys, features = set(), [], []
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=output.parent, suffix=".csv.tmp", delete=False,
                                         newline="") as handle:
            temporary = handle.name
            writer = csv.writer(handle)
            writer.writerow(["transcript_id", "transcript_position", "score"])

            def write_batch():
                scores = pipeline.predict_proba(np.asarray(features))[:, 1]
                if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                    raise ValueError("Invalid predicted probabilities")
                for (transcript, position), score in zip(keys, scores):
                    writer.writerow([transcript, position, float(score)])
                keys.clear()
                features.clear()

            for transcript, position, x in iter_sites(data):
                key = (transcript, position)
                if key in seen:
                    raise ValueError(f"Duplicate input site: {key}")
                seen.add(key)
                keys.append(key)
                features.append(x)
                if len(keys) >= batch_size:
                    write_batch()
            if keys:
                write_batch()
            if not seen:
                raise ValueError("No sites found")
        os.replace(temporary, output)
    except Exception:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "models/baseline_model.joblib"),
                        help="Defaults to the packaged data0 + data1 + data2 model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    count = predict(args.data, args.model, args.output, args.batch_size)
    print(f"Wrote {count:,} site probabilities to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
