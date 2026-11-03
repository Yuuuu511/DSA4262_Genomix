
"""
SG-NEx Differential Prediction Atlas — multi-threshold version

Reads existing consensus_*.parquet files.
Builds complete atlases for positive thresholds:
    0.3, 0.5, 0.7, 0.9

Base differential threshold:
    abs(mean_score_A - mean_score_B) >= 0.3

Run:
    python run_differential.py --output outputs

No XGBoost inference or raw prediction CSV loading required.
"""

from pathlib import Path
from itertools import combinations
import argparse
import gc
import json

import numpy as np
import pandas as pd

from task2_pipeline import build_differential_pair

POS_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
BASE_DELTA = 0.3
EPS = 1e-7  # Preserve the audited float32 boundary convention.
SITE_KEY = ["transcript_id", "transcript_position"]


def candidate_mask(d, positive_threshold, delta_threshold):
    """Canonical atlas selection, including its audited float32 tolerance."""
    return (
        (d["abs_delta_mean"] >= delta_threshold - EPS)
        & (d["higher_mean_score"] >= positive_threshold - EPS)
    )


def tag(t):
    return f"{float(t):.1f}"


def rate(n, d):
    return n / d if d else np.nan


def transcript_summary(df):
    cols = [
        "transcript_id", "n_diff_sites",
        "n_replicate_supported",
        "max_abs_delta", "mean_abs_delta",
        "major_direction",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    result = (
        df.groupby("transcript_id", as_index=False)
        .agg(
            n_diff_sites=("transcript_position", "size"),
            n_replicate_supported=(
                "replicate_supported", "sum"
            ),
            max_abs_delta=("abs_delta_mean", "max"),
            mean_abs_delta=("abs_delta_mean", "mean"),
            major_direction=(
                "higher_cell",
                lambda s: s.value_counts().idxmax()
            ),
        )
    )
    return result.sort_values(
        ["n_replicate_supported", "n_diff_sites", "max_abs_delta"],
        ascending=False,
    )


def summarize_pair(d, selected, a, b, threshold):
    strict = selected[selected["replicate_supported"]]

    n_a = int((selected["higher_cell"] == a).sum())
    n_b = int((selected["higher_cell"] == b).sum())
    s_a = int((strict["higher_cell"] == a).sum())
    s_b = int((strict["higher_cell"] == b).sum())

    return {
        "cell_a": a,
        "cell_b": b,
        "pair": f"{a}_vs_{b}",
        "positive_threshold": threshold,
        "base_delta_threshold": BASE_DELTA,
        "n_matched_sites": len(d),
        "n_selected": len(selected),
        "n_strict": len(strict),
        "selected_fraction": rate(len(selected), len(d)),
        "strict_fraction_of_matched": rate(len(strict), len(d)),
        "strict_support_rate": rate(len(strict), len(selected)),
        "n_higher_a": n_a,
        "n_higher_b": n_b,
        "n_strict_higher_a": s_a,
        "n_strict_higher_b": s_b,
        "strict_support_rate_a": rate(s_a, n_a),
        "strict_support_rate_b": rate(s_b, n_b),
        "n_differential_transcripts": (
            selected["transcript_id"].nunique()
        ),
        "n_strict_transcripts": (
            strict["transcript_id"].nunique()
        ),
    }


def save_atlas_tables(frames, folder, threshold, all_cells):
    folder.mkdir(parents=True, exist_ok=True)

    summaries = pd.DataFrame(frames["summary"])
    summaries.to_csv(
        folder / "differential_pair_summary.csv",
        index=False
    )

    if frames["sites"]:
        sites = pd.concat(frames["sites"], ignore_index=True)
    else:
        sites = pd.DataFrame()

    sites.to_parquet(
        folder / "differential_site_atlas.parquet",
        index=False
    )

    # CSV retained for convenient inspection / compatibility.
    sites.to_csv(
        folder / "differential_site_atlas.csv",
        index=False
    )

    strict = (
        sites[sites["replicate_supported"]].copy()
        if not sites.empty else sites.copy()
    )
    strict.to_csv(
        folder / "differential_strict_atlas.csv",
        index=False
    )

    transcripts = (
        pd.concat(frames["transcripts"], ignore_index=True)
        if frames["transcripts"] else pd.DataFrame()
    )
    transcripts.to_csv(
        folder / "differential_transcript_atlas.csv",
        index=False
    )

    matrix = pd.DataFrame(
        np.nan, index=all_cells, columns=all_cells
    )
    for _, r in summaries.iterrows():
        a, b = r["cell_a"], r["cell_b"]
        matrix.loc[a, b] = r["selected_fraction"]
        matrix.loc[b, a] = r["selected_fraction"]

    matrix.index.name = "cell_line"
    matrix.to_csv(folder / "differential_pair_matrix.csv")

    # Direction-preserving recurrence: a site can appear
    # multiple times across pairwise comparisons.
    if not sites.empty:
        grouped = (
            sites.groupby(SITE_KEY, as_index=False)
            .agg(
                n_pair_comparisons=("pair", "nunique"),
                n_strict_comparisons=(
                    "replicate_supported", "sum"
                ),
                max_abs_delta=("abs_delta_mean", "max"),
            )
            .sort_values(
                ["n_strict_comparisons", "n_pair_comparisons"],
                ascending=False
            )
        )
        grouped.to_csv(
            folder / "recurring_differential_sites.csv",
            index=False
        )

        higher = (
            sites.groupby(SITE_KEY + ["higher_cell"])
            .size()
            .rename("n_higher_comparisons")
            .reset_index()
        )
        higher.to_csv(
            folder / "site_enrichment_directions.csv",
            index=False
        )

    print(
        f"\nThreshold {threshold:.1f}: "
        f"{len(sites):,} selected, "
        f"{len(strict):,} strict pair-site records"
    )
    return summaries


def build_all(output, min_runs):
    tables = output / "tables"
    root = tables / "differential_atlas"

    cells = sorted(
        p.stem.replace("consensus_", "", 1)
        for p in tables.glob("consensus_*.parquet")
    )
    if len(cells) < 2:
        raise FileNotFoundError(
            "Need at least two consensus_*.parquet files."
        )

    pairs = list(combinations(cells, 2))

    buckets = {
        t: {"summary": [], "sites": [], "transcripts": []}
        for t in POS_THRESHOLDS
    }

    print("SG-NEx Differential Prediction Atlas")
    print("Cell lines:", cells)
    print("Pairs:", len(pairs))
    print("Positive thresholds:", POS_THRESHOLDS)
    print("Base differential threshold:", BASE_DELTA)
    print("Minimum runs per cell:", min_runs)

    for i, (a, b) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {a} vs {b}", flush=True)

        # Build matched-site data ONCE per pair.
        # d is not selected by positive threshold.
        d, _, _ = build_differential_pair(
            tables=tables,
            cell_a=a,
            cell_b=b,
            min_runs=min_runs,
            delta_threshold=BASE_DELTA,
        )

        # Store only scores already passing base delta;
        # each positive-threshold atlas filters this subset.
        base = d[
            d["abs_delta_mean"] >= BASE_DELTA - EPS
        ].copy()

        for t in POS_THRESHOLDS:
            selected = base[candidate_mask(base, t, BASE_DELTA)].copy()

            # These rows belong to this folder's candidate set. The
            # pipeline's inherited flag instead refers to positive >= 0.5.
            selected["selected"] = candidate_mask(selected, t, BASE_DELTA)
            selected["positive_threshold"] = t
            selected["base_delta_threshold"] = BASE_DELTA
            # Consensus positive-run counts were computed at 0.5, even
            # when the atlas's higher-consensus-score threshold differs.
            selected = selected.rename(columns={
                f"n_positive_runs_{cell}": f"n_positive_runs_at_0_5_{cell}"
                for cell in (a, b)
            })

            selected.insert(0, "pair", f"{a}_vs_{b}")
            selected.insert(1, "cell_a", a)
            selected.insert(2, "cell_b", b)

            buckets[t]["summary"].append(
                summarize_pair(d, selected, a, b, t)
            )

            if not selected.empty:
                buckets[t]["sites"].append(selected)

                tx = transcript_summary(selected)
                tx.insert(0, "pair", f"{a}_vs_{b}")
                tx.insert(1, "cell_a", a)
                tx.insert(2, "cell_b", b)
                buckets[t]["transcripts"].append(tx)

            strict_n = int(
                selected["replicate_supported"].sum()
            )
            print(
                f"  positive >= {t:.1f}: "
                f"selected={len(selected):,}, strict={strict_n:,}"
            )

            del selected

        del d, base
        gc.collect()

    for t in POS_THRESHOLDS:
        folder = root / f"positive_{tag(t)}"
        save_atlas_tables(
            buckets[t], folder, t, cells
        )

    # Metadata lets dashboard verify compatibility.
    metadata = {
        "schema_version": 2,
        "positive_thresholds": list(POS_THRESHOLDS),
        "base_delta_threshold": BASE_DELTA,
        "min_detected_runs": min_runs,
        "cell_lines": cells,
        "n_pairs": len(pairs),
        "comparison_tolerance": EPS,
        "positive_run_count_threshold": 0.5,
        "field_definitions": {
            "selected": (
                "Meets this row's positive_threshold and base_delta_threshold "
                "within comparison_tolerance; true for every exported candidate."
            ),
            "n_positive_runs_at_0_5_<cell>": (
                "Observed runs with score >= 0.5; independent of atlas folder."
            ),
            "replicate_supported": (
                "Strict separation of observed run-score ranges, not a statistical test."
            ),
            "site_key": "Normalized transcript_id and transcript_position; not a genomic locus.",
        },
        "description": (
            "XGBoost differential prediction candidates; "
            "not statistically validated differential methylation."
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "atlas_manifest.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nCompleted. Atlas root:", root.resolve())
    print("All four positive thresholds available.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("outputs")
    )
    parser.add_argument(
        "--min-runs", type=int, default=2
    )
    args = parser.parse_args()

    if args.min_runs < 1:
        parser.error("--min-runs must be >= 1")

    build_all(args.output.expanduser(), args.min_runs)


if __name__ == "__main__":
    main()
