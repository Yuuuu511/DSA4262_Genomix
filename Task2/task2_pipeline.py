
"""
DSA4262 Task 2 — SG-NEx m6A Analysis
=====================================

Input:
    sg_nex_prediction_results/
        A549/A549_rep5_run1.csv
        ...
        K562/K562_rep6_run1.csv

Each prediction CSV:
    transcript_id,transcript_position,score

Output:
    outputs/tables/
    outputs/figures/

Main analyses:
    1. Prediction statistics
    2. Replicate reproducibility
    3. Threshold sensitivity
    4. Cross-cell-line similarity
    5. Candidate-site sharing
    6. Differential prediction candidates
    7. Transcript-level exploration

No additional genomic annotation required.

Important:
    Ensembl version suffixes are removed for matching.
    Coordinate compatibility between transcript versions
    is assumed, not independently verified.

    Predicted-positive sites are not experimentally
    validated m6A modifications.
"""

from pathlib import Path
import argparse
import itertools
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

KEY = ["transcript_id", "transcript_position"]

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
MAIN_T = 0.5
MIN_RUNS = 2

COLORS = [
    "#4477AA",
    "#EE8866",
    "#228833",
    "#AA3377",
    "#66CCEE",
    "#CCBB44",
    "#999999",
]

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# GENERAL HELPERS
# ============================================================

def savefig(name, folder):
    folder.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "pdf", "svg"):
        plt.savefig(
            folder / f"{name}.{ext}",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close()


def nice_limits(values, min_span=0.03, pad_frac=0.08,
                lower_clip=0.0, upper_clip=1.0):
    s = pd.Series(values).dropna().astype(float)

    if s.empty:
        return lower_clip, upper_clip

    lo = float(s.min())
    hi = float(s.max())

    span = max(hi - lo, min_span)
    midpoint = (hi + lo) / 2

    half = max((hi - lo) / 2, min_span / 2)
    half *= 1 + 2 * pad_frac

    return (
        max(lower_clip, midpoint - half),
        min(upper_clip, midpoint + half)
    )


def setup_style():
    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.20
    )


def get_palette(cells):
    return dict(zip(
        cells,
        (COLORS * (len(cells) // len(COLORS) + 1))[:len(cells)]
    ))


# ============================================================
# DATA LOADING
# ============================================================

def read_prediction(path):
    df = pd.read_csv(
        path,
        usecols=KEY + ["score"],
        dtype={
            "transcript_id": "string",
            "transcript_position": "int32",
            "score": "float32"
        }
    )

    if df[KEY].isna().any().any():
        raise ValueError(f"Missing site identifier: {path}")

    if df["score"].isna().any():
        raise ValueError(f"Missing prediction score: {path}")

    if not df["score"].between(0, 1).all():
        raise ValueError(f"Scores outside [0,1]: {path}")

    # ENST00000000233.9 -> ENST00000000233
    df["transcript_id"] = df["transcript_id"].str.replace(
        r"^(ENST\d+)\.\d+$",
        r"\1",
        regex=True
    )

    duplicates = int(df.duplicated(KEY).sum())

    if duplicates:
        raise ValueError(
            f"{path}: {duplicates} duplicate site keys "
            "after transcript version normalization."
        )

    return df


def discover(root):
    files = {}

    pattern = re.compile(
        r"^(.+)_rep\d+_run\d+\.csv$"
    )

    for p in sorted(root.glob("*/*.csv")):
        match = pattern.match(p.name)

        if match and match.group(1).lower() == p.parent.name.lower():
            files.setdefault(p.parent.name, []).append(p)

    if not files:
        raise ValueError(
            f"No valid prediction CSVs found under {root}"
        )

    return files


# ============================================================
# REPLICATE COMPARISON
# ============================================================

def pairs_similarity(a, b, cell, run_a, run_b):
    x = a.merge(
        b,
        on=KEY,
        how="inner",
        suffixes=("_a", "_b")
    )

    n = len(x)

    s1 = x["score_a"]
    s2 = x["score_b"]

    result = {
        "cell_line": cell,
        "run_a": run_a,
        "run_b": run_b,
        "n_a": len(a),
        "n_b": len(b),
        "n_shared": n,
        "fraction_a": n / len(a),
        "fraction_b": n / len(b),
        "pearson": s1.corr(s2) if n > 2 else np.nan,
        "spearman": (
            s1.corr(s2, method="spearman")
            if n > 2 else np.nan
        )
    }

    for t in THRESHOLDS:
        p = s1 >= t
        q = s2 >= t

        both = int((p & q).sum())
        union = int((p | q).sum())

        result[f"jaccard_{t}"] = (
            both / union if union else np.nan
        )
        result[f"both_positive_{t}"] = both
        result[f"positive_a_{t}"] = int(p.sum())
        result[f"positive_b_{t}"] = int(q.sum())

    return result


# ============================================================
# CELL-LINE CONSENSUS
# ============================================================

def consensus_for_cell(cell, paths, tables):
    runs = []
    audit = []

    for path in paths:
        print("Reading", path, flush=True)

        df = read_prediction(path)
        run = path.stem

        row = {
            "cell_line": cell,
            "run": run,
            "n_sites": len(df),
            "n_transcripts": df.transcript_id.nunique(),
            "mean_score": float(df.score.mean()),
            "median_score": float(df.score.median()),
            "score_sd": float(df.score.std()),
            "non_ENST_sites": int(
                (~df.transcript_id.str.startswith("ENST")).sum()
            )
        }

        for t in THRESHOLDS:
            row[f"positive_{t}"] = int((df.score >= t).sum())

        audit.append(row)
        runs.append((run, df))

    pairwise = [
        pairs_similarity(a, b, cell, ra, rb)
        for (ra, a), (rb, b)
        in itertools.combinations(runs, 2)
    ]

    joined = pd.concat(
        [df for _, df in runs],
        ignore_index=True
    )

    joined["positive"] = (
        joined["score"] >= MAIN_T
    ).astype("int8")

    agg = joined.groupby(KEY, sort=False).agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        n_detected_runs=("score", "size"),
        n_positive_runs=("positive", "sum")
    ).reset_index()

    agg["std_score"] = agg["std_score"].fillna(0)
    agg["n_total_runs"] = len(runs)

    agg["positive_consensus"] = (
        agg.mean_score >= MAIN_T
    )

    agg["all_observed_positive"] = (
        agg.n_positive_runs == agg.n_detected_runs
    )

    agg["stable_positive"] = (
        (agg.n_detected_runs >= MIN_RUNS)
        & agg.positive_consensus
        & agg.all_observed_positive
    )

    agg["cell_line"] = cell

    agg.to_parquet(
        tables / f"consensus_{cell}.parquet",
        index=False
    )

    coverage = (
        agg.n_detected_runs.value_counts()
        .sort_index()
        .rename_axis("n_detected_runs")
        .reset_index(name="n_sites")
    )

    coverage["cell_line"] = cell
    coverage["n_total_runs"] = len(runs)

    sensitivity = []

    for t in THRESHOLDS:
        for minruns in sorted({1, 2, len(runs)}):
            subset = agg[
                agg.n_detected_runs >= minruns
            ]

            positive = subset.mean_score >= t

            sensitivity.append({
                "cell_line": cell,
                "threshold": t,
                "min_runs": minruns,
                "n_evaluable": len(subset),
                "n_positive": int(positive.sum()),
                "positive_fraction": (
                    float(positive.mean())
                    if len(subset) else np.nan
                )
            })

    del joined

    return audit, pairwise, coverage, sensitivity


# ============================================================
# CROSS-CELL-LINE COMPARISON
# ============================================================

def cross_cell_analysis(cells, tables):
    wide = None

    for cell in cells:
        d = pd.read_parquet(
            tables / f"consensus_{cell}.parquet",
            columns=KEY + [
                "mean_score",
                "n_detected_runs"
            ]
        )

        d = d[d.n_detected_runs >= MIN_RUNS]

        d = d[KEY + ["mean_score"]].rename(
            columns={"mean_score": cell}
        )

        if wide is None:
            wide = d
        else:
            wide = wide.merge(
                d,
                on=KEY,
                how="outer",
                sort=False
            )

    wide.to_parquet(
        tables / "cross_cell_scores.parquet",
        index=False
    )

    records = []

    for a, b in itertools.combinations(cells, 2):
        matched = wide[[a, b]].dropna()

        if matched.empty:
            continue

        x = matched[a]
        y = matched[b]

        row = {
            "cell_a": a,
            "cell_b": b,
            "n_shared": len(matched),
            "pearson": x.corr(y),
            "spearman": x.corr(y, method="spearman"),
            "mean_delta": float((x - y).mean())
        }

        for t in THRESHOLDS:
            p = x >= t
            q = y >= t

            union = int((p | q).sum())
            both = int((p & q).sum())

            row[f"positive_jaccard_{t}"] = (
                both / union if union else np.nan
            )
            row[f"shared_positive_{t}"] = both

        records.append(row)

    cross = pd.DataFrame(records)

    cross.to_csv(
        tables / "cross_cell_pairwise.csv",
        index=False
    )

    site_n = wide[cells].notna().sum(axis=1)
    site_pos = wide[cells].ge(MAIN_T).sum(axis=1)

    sharing = (
        pd.DataFrame({
            "n_cell_lines_observed": site_n,
            "n_cell_lines_positive": site_pos
        })
        .value_counts()
        .rename("n_sites")
        .reset_index()
    )

    sharing.to_csv(
        tables / "site_sharing.csv",
        index=False
    )

    return cross


# ============================================================
# DIFFERENTIAL PREDICTION ANALYSIS
# ============================================================

def build_differential_pair(
    tables,
    cell_a,
    cell_b,
    min_runs=2,
    delta_threshold=0.3
):
    keep = [
        "mean_score",
        "std_score",
        "min_score",
        "max_score",
        "n_detected_runs",
        "n_positive_runs"
    ]

    a = pd.read_parquet(
        tables / f"consensus_{cell_a}.parquet",
        columns=KEY + keep
    )

    b = pd.read_parquet(
        tables / f"consensus_{cell_b}.parquet",
        columns=KEY + keep
    )

    a = a.rename(
        columns={c: f"{c}_{cell_a}" for c in keep}
    )

    b = b.rename(
        columns={c: f"{c}_{cell_b}" for c in keep}
    )

    d = a.merge(b, on=KEY, how="inner")

    # Use the same evaluable-site condition as cross-cell analysis
    d = d[
        (d[f"n_detected_runs_{cell_a}"] >= min_runs)
        & (d[f"n_detected_runs_{cell_b}"] >= min_runs)
    ].copy()

    d["delta_mean"] = (
        d[f"mean_score_{cell_a}"]
        - d[f"mean_score_{cell_b}"]
    )

    d["abs_delta_mean"] = d.delta_mean.abs()

    d["higher_cell"] = np.where(
        d.delta_mean >= 0,
        cell_a,
        cell_b
    )

    d["higher_mean_score"] = np.where(
        d.delta_mean >= 0,
        d[f"mean_score_{cell_a}"],
        d[f"mean_score_{cell_b}"]
    )

    # Conservative separation of observed replicate ranges
    d["conservative_gap"] = np.where(
        d.delta_mean >= 0,
        d[f"min_score_{cell_a}"]
        - d[f"max_score_{cell_b}"],
        d[f"min_score_{cell_b}"]
        - d[f"max_score_{cell_a}"]
    )

    d["replicate_supported"] = (
        d.conservative_gap > 0
    )

    d["selected"] = (
        (d.abs_delta_mean >= delta_threshold)
        & (d.higher_mean_score >= MAIN_T)
    )

    selected = d[d.selected].copy()

    strict = selected[
        selected.replicate_supported
    ].copy()

    selected = selected.sort_values(
        "abs_delta_mean",
        ascending=False
    )

    strict = strict.sort_values(
        "abs_delta_mean",
        ascending=False
    )

    return d, selected, strict


def summarize_differential_transcripts(df):
    columns = [
        "transcript_id",
        "n_diff_sites",
        "n_replicate_supported",
        "max_abs_delta",
        "mean_abs_delta",
        "major_direction"
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    result = (
        df.groupby("transcript_id")
        .agg(
            n_diff_sites=("transcript_position", "size"),
            n_replicate_supported=(
                "replicate_supported", "sum"
            ),
            max_abs_delta=("abs_delta_mean", "max"),
            mean_abs_delta=("abs_delta_mean", "mean"),
            major_direction=(
                "higher_cell",
                lambda x: x.value_counts().idxmax()
            )
        )
        .reset_index()
    )

    return result.sort_values(
        [
            "n_replicate_supported",
            "n_diff_sites",
            "max_abs_delta"
        ],
        ascending=False
    )


# ============================================================
# FIGURES 1–3
# ============================================================

def make_figures(tables, figs, cells, pairwise, cross):
    setup_style()
    palette = get_palette(cells)

    overview = pd.read_csv(
        tables / "prediction_overview.csv"
    )

    # --------------------------------------------------------
    # FIGURE 1 — Positive prediction fractions
    # --------------------------------------------------------
    overview["positive_fraction_pct"] = (
        100 * overview["positive_0.5"] / overview["n_sites"]
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))

    sns.barplot(
        data=overview,
        x="cell_line",
        y="positive_fraction_pct",
        hue="cell_line",
        palette=palette,
        legend=False,
        errorbar=None,
        alpha=0.40,
        ax=ax
    )

    sns.stripplot(
        data=overview,
        x="cell_line",
        y="positive_fraction_pct",
        color="#303030",
        size=5,
        alpha=0.70,
        jitter=False,
        ax=ax
    )

    ax.set(
        xlabel="Cell line",
        ylabel="Predicted-positive candidate sites (%)",
        title="Predicted-positive fractions (score ≥ 0.5)"
    )
    ax.set_ylim(bottom=0)

    sns.despine()
    savefig("fig1_prediction_overview", figs)

    # --------------------------------------------------------
    # FIGURE 2A — Replicate Pearson
    # NO horizontal jitter
    # One dot represents one run pair
    # --------------------------------------------------------
    if not pairwise.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))

        sns.stripplot(
            data=pairwise,
            x="cell_line",
            y="pearson",
            hue="cell_line",
            palette=palette,
            jitter=False,
            size=5,
            alpha=0.50,
            legend=False,
            ax=ax
        )

        y0, y1 = nice_limits(
            pairwise.pearson,
            min_span=0.05
        )

        ax.set(
            xlabel="Cell line",
            ylabel="Pearson r on shared candidate sites",
            ylim=(y0, y1),
            title="Within-cell-line run-pair reproducibility"
        )

        sns.despine()
        savefig("fig2_replicate_correlation", figs)

        # ----------------------------------------------------
        # FIGURE 2B — Mean Jaccard ONLY
        # No background individual-pair dots
        # ----------------------------------------------------
        long = pairwise.melt(
            id_vars=["cell_line", "run_a", "run_b"],
            value_vars=[
                f"jaccard_{t}" for t in THRESHOLDS
            ],
            var_name="threshold",
            value_name="jaccard"
        )

        long["threshold"] = (
            long.threshold
            .str.replace("jaccard_", "", regex=False)
            .astype(float)
        )

        mean_jaccard = (
            long.groupby(
                ["cell_line", "threshold"],
                as_index=False
            )["jaccard"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(8.5, 4.8))

        sns.lineplot(
            data=mean_jaccard,
            x="threshold",
            y="jaccard",
            hue="cell_line",
            hue_order=cells,
            palette=palette,
            marker="o",
            markersize=5,
            linewidth=2.0,
            errorbar=None,
            ax=ax
        )

        y0, y1 = nice_limits(
            mean_jaccard.jaccard,
            min_span=0.08
        )

        ax.set(
            xlabel="Prediction threshold",
            ylabel="Mean positive-site Jaccard similarity",
            ylim=(y0, y1),
            title="Run-pair positive overlap within shared observations"
        )

        ax.set_xticks(THRESHOLDS)

        ax.legend(
            title="Cell line",
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        sns.despine()
        savefig("fig2_threshold_sensitivity", figs)

    # --------------------------------------------------------
    # FIGURE 3A — Cross-cell Pearson heatmap
    # --------------------------------------------------------
    pearson_mat = pd.DataFrame(
        np.nan,
        index=cells,
        columns=cells,
        dtype=float
    )

    for row in cross.itertuples():
        pearson_mat.loc[row.cell_a, row.cell_b] = row.pearson
        pearson_mat.loc[row.cell_b, row.cell_a] = row.pearson

    mask = np.eye(len(cells), dtype=bool)

    p0, p1 = nice_limits(
        pearson_mat.stack(),
        min_span=0.04
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    sns.heatmap(
        pearson_mat,
        mask=mask,
        vmin=p0,
        vmax=p1,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Pearson r"},
        ax=ax
    )

    ax.set(
        xlabel="",
        ylabel="",
        title="Cross-cell score similarity on shared candidates"
    )
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    savefig("fig3_cross_cell_similarity", figs)

    # --------------------------------------------------------
    # FIGURE 3B — Cross-cell positive Jaccard heatmap
    # --------------------------------------------------------
    jaccard_mat = pd.DataFrame(
        np.nan,
        index=cells,
        columns=cells,
        dtype=float
    )

    for _, row in cross.iterrows():
        value = row["positive_jaccard_0.5"]

        jaccard_mat.loc[row["cell_a"], row["cell_b"]] = value
        jaccard_mat.loc[row["cell_b"], row["cell_a"]] = value

    j0, j1 = nice_limits(
        jaccard_mat.stack(),
        min_span=0.05
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    sns.heatmap(
        jaccard_mat,
        mask=mask,
        vmin=j0,
        vmax=j1,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Positive-site Jaccard"},
        ax=ax
    )

    ax.set(
        xlabel="",
        ylabel="",
        title="Cross-cell-line positive-site overlap (score ≥ 0.5)"
    )

    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    savefig("fig3_cross_cell_jaccard", figs)

    # --------------------------------------------------------
    # Candidate-site sharing
    # This describes detection, NOT modification specificity.
    # --------------------------------------------------------
    sharing = pd.read_csv(
        tables / "site_sharing.csv"
    )

    observed = (
        sharing.groupby(
            "n_cell_lines_observed",
            as_index=False
        )["n_sites"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    sns.barplot(
        data=observed,
        x="n_cell_lines_observed",
        y="n_sites",
        color="#4477AA",
        ax=ax
    )

    ax.set(
        xlabel="Cell lines detecting the site in ≥2 runs",
        ylabel="Candidate sites",
        title="Candidate-site detection across cell lines"
    )

    sns.despine()
    savefig("figS1_site_sharing", figs)


# ============================================================
# FIGURE 4 — DIFFERENTIAL PREDICTION
# ============================================================

def make_figure4_differential(
    tables,
    figs,
    cell_a,
    cell_b,
    min_runs=2,
    delta_threshold=0.3
):
    d, selected, strict = build_differential_pair(
        tables,
        cell_a,
        cell_b,
        min_runs,
        delta_threshold
    )

    tx = summarize_differential_transcripts(selected)

    stem = f"{cell_a}_vs_{cell_b}"

    d.to_csv(
        tables / f"differential_all_{stem}.csv",
        index=False
    )
    selected.to_csv(
        tables / f"differential_selected_{stem}.csv",
        index=False
    )
    strict.to_csv(
        tables / f"differential_strict_{stem}.csv",
        index=False
    )
    tx.to_csv(
        tables / f"differential_transcripts_{stem}.csv",
        index=False
    )

    print(f"\nDifferential analysis: {cell_a} vs {cell_b}")
    print("Matched evaluable sites:", len(d))
    print("Selected differential candidates:", len(selected))
    print("Replicate-supported candidates:", len(strict))

    if len(selected):
        print(
            "Strict support fraction:",
            round(len(strict) / len(selected), 4)
        )
        print("\nBy direction:")
        print(
            selected.groupby("higher_cell")
            .agg(
                n_selected=("transcript_id", "size"),
                n_strict=("replicate_supported", "sum")
            )
            .to_string()
        )

    if d.empty:
        print("No comparable sites. Skipping Figure 4.")
        return

    setup_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13.5, 10),
        constrained_layout=True
    )

    # --------------------------------------------------------
    # PANEL A — Logarithmic hexbin density
    # --------------------------------------------------------
    hb = axes[0, 0].hexbin(
        d[f"mean_score_{cell_a}"],
        d[f"mean_score_{cell_b}"],
        gridsize=75,
        mincnt=1,
        bins="log",
        cmap="Blues"
    )

    axes[0, 0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        color="gray"
    )

    axes[0, 0].set(
        xlabel=f"{cell_a} mean prediction score",
        ylabel=f"{cell_b} mean prediction score",
        title="A  Matched-site prediction scores",
        xlim=(0, 1),
        ylim=(0, 1)
    )

    fig.colorbar(
        hb,
        ax=axes[0, 0],
        label="Site count (log scale)"
    )

    # --------------------------------------------------------
    # PANEL B — Difference distribution
    # --------------------------------------------------------
    axes[0, 1].hist(
        d.delta_mean,
        bins=120,
        color="#4477AA"
    )

    axes[0, 1].set_yscale("log")

    axes[0, 1].axvline(
        0,
        linestyle="--",
        linewidth=1,
        color="gray"
    )

    for value in [-delta_threshold, delta_threshold]:
        axes[0, 1].axvline(
            value,
            linestyle="--",
            linewidth=1,
            color="#AA3377"
        )

    axes[0, 1].set(
        xlabel=f"Mean score difference ({cell_a} − {cell_b})",
        ylabel="Candidate sites (log scale)",
        title="B  Distribution of prediction differences"
    )

    # --------------------------------------------------------
    # PANEL C — Differential transcript ranking
    # --------------------------------------------------------
    ax = axes[1, 0]

    top_tx = tx.head(12).copy()

    if len(top_tx):
        top_tx = top_tx.sort_values(
            [
                "n_replicate_supported",
                "n_diff_sites"
            ],
            ascending=True
        )

        positions = np.arange(len(top_tx))

        ax.barh(
            positions,
            top_tx.n_replicate_supported,
            color="#228833",
            label="Strict-supported sites"
        )

        ax.scatter(
            top_tx.n_diff_sites,
            positions,
            color="#AA3377",
            s=24,
            zorder=3,
            label="All selected sites"
        )

        ax.set_yticks(
            positions,
            top_tx.transcript_id
        )

        ax.legend(
            frameon=False,
            loc="lower right",
            fontsize=8
        )
    else:
        ax.text(
            0.5, 0.5,
            "No differential transcripts found",
            transform=ax.transAxes,
            ha="center", va="center"
        )

    ax.set(
        xlabel="Number of differential prediction sites",
        ylabel="Transcript ID",
        title="C  Top differential transcripts"
    )

    # --------------------------------------------------------
    # PANEL D — Discrete transcript-position predictions
    # --------------------------------------------------------
    ax = axes[1, 1]

    if len(tx):
        chosen = str(tx.iloc[0].transcript_id)

        dd = (
            d[d.transcript_id == chosen]
            .sort_values("transcript_position")
        )

        selected_tx = selected[
            selected.transcript_id == chosen
        ]

        for cell, color in [
            (cell_a, "#4477AA"),
            (cell_b, "#EE8866")
        ]:
            ax.scatter(
                dd.transcript_position,
                dd[f"mean_score_{cell}"],
                color=color,
                s=24,
                alpha=0.75,
                label=cell,
                zorder=2
            )

            # Hollow rings highlight selected differential sites
            ax.scatter(
                selected_tx.transcript_position,
                selected_tx[f"mean_score_{cell}"],
                s=68,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                label="Selected site (hollow ring)" if cell == cell_a else None,
                zorder=4
            )

        ax.axhline(
            MAIN_T,
            color="gray",
            linewidth=1,
            linestyle="--",
            label=f"Threshold = {MAIN_T}"
        )

        ax.set(
            xlabel="Transcript position",
            ylabel="Mean prediction score",
            title=f"D  Transcript {chosen}",
            ylim=(0, 1)
        )

        ax.legend(
            frameon=False,
            fontsize=8
        )

    else:
        ax.text(
            0.5, 0.5,
            "No transcript available",
            transform=ax.transAxes,
            ha="center",
            va="center"
        )
        ax.set_axis_off()

    savefig(
        f"fig4_differential_{stem}",
        figs
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path
    )
    parser.add_argument(
        "--output",
        default=Path("outputs"),
        type=Path
    )
    parser.add_argument(
        "--focus-a",
        default="Hct116"
    )
    parser.add_argument(
        "--focus-b",
        default="K562"
    )
    parser.add_argument(
        "--delta-threshold",
        default=0.3,
        type=float
    )

    args = parser.parse_args()

    root = args.input.expanduser()
    out = args.output.expanduser()

    tables = out / "tables"
    figs = out / "figures"

    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    files = discover(root)
    cells = list(files)

    if args.focus_a not in cells or args.focus_b not in cells:
        parser.error(
            f"Focus cells must be in {cells}"
        )

    if args.focus_a == args.focus_b:
        parser.error("Focus cell lines must differ.")

    print("Cell lines:", cells)
    print("Total runs:", sum(len(x) for x in files.values()))

    all_audit = []
    all_pairs = []
    all_coverage = []
    all_sensitivity = []

    for cell, paths in files.items():
        audit, pairs, coverage, sensitivity = (
            consensus_for_cell(cell, paths, tables)
        )

        all_audit.extend(audit)
        all_pairs.extend(pairs)
        all_coverage.append(coverage)
        all_sensitivity.extend(sensitivity)

    overview = pd.DataFrame(all_audit)
    pairwise = pd.DataFrame(all_pairs)

    overview.to_csv(
        tables / "prediction_overview.csv",
        index=False
    )
    pairwise.to_csv(
        tables / "replicate_pairwise.csv",
        index=False
    )

    pd.concat(
        all_coverage,
        ignore_index=True
    ).to_csv(
        tables / "replicate_detection_coverage.csv",
        index=False
    )

    pd.DataFrame(all_sensitivity).to_csv(
        tables / "threshold_sensitivity.csv",
        index=False
    )

    cross = cross_cell_analysis(
        cells,
        tables
    )

    make_figures(
        tables,
        figs,
        cells,
        pairwise,
        cross
    )

    make_figure4_differential(
        tables=tables,
        figs=figs,
        cell_a=args.focus_a,
        cell_b=args.focus_b,
        min_runs=MIN_RUNS,
        delta_threshold=args.delta_threshold
    )

    print("\nCompleted:", out.resolve())

    print("\nRun overview:")
    print(
        overview.groupby("cell_line")[
            ["n_sites", "positive_0.5"]
        ].mean().round(1)
    )

    print("\nCross-cell comparisons:", len(cross))


if __name__ == "__main__":
    main()
