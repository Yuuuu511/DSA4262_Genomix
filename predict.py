"""Stream new sites and return the required three-column predictions."""
import argparse
import csv
import joblib
import pandas as pd
from features import iter_sites, summarize_site
from normalization import normalized_site


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    saved = joblib.load(args.model)
    reference = saved.get("reference")
    buffer, seen = [], set()
    def flush(writer):
        if not buffer:
            return
        frame = pd.DataFrame(buffer)
        scores = saved["pipeline"].predict_proba(frame[saved["columns"]])[:, 1]
        for row, score in zip(buffer, scores):
            writer.writerow([row["transcript_id"], row["transcript_position"], float(score)])
        buffer.clear()
    with open(args.output, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["transcript_id", "transcript_position", "score"])
        for transcript, pos, seq, reads in iter_sites(args.json):
            key = (transcript, pos)
            if key in seen:
                raise ValueError(f"Duplicate input site: {key}")
            seen.add(key)
            row = {"transcript_id": transcript, "transcript_position": pos,
                   **summarize_site(reads, seq)}
            if reference is not None:
                row.update(normalized_site(reads, seq, reference))
            buffer.append(row)
            if len(buffer) >= 1000:
                flush(writer)
        flush(writer)
    print(f"Wrote {len(seen)} site predictions")


if __name__ == "__main__":
    main()
