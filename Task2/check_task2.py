
import sys

if "--qc-tests" in sys.argv:
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName("test_quality_control")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

import pandas as pd
from itertools import combinations

# Load analysis outputs
overview = pd.read_csv("outputs/tables/prediction_overview.csv")
cross = pd.read_csv("outputs/tables/cross_cell_pairwise.csv")

# ==========================================
# CHECK 1: Why 15 comparisons instead of 21?
# ==========================================

cells = sorted(overview["cell_line"].unique())

existing = {
    tuple(sorted([a, b]))
    for a, b in zip(cross.cell_a, cross.cell_b)
}

missing = [
    (a, b)
    for a, b in combinations(cells, 2)
    if (a, b) not in existing
]

print("\n=== MISSING CELL-LINE COMPARISONS ===")
for pair in missing:
    print(pair)

print("\n=== AVAILABLE CELL-LINE COMPARISONS ===")
print(
    cross[["cell_a", "cell_b", "n_shared",
           "pearson", "mean_delta"]].to_string(index=False)
)

# ==========================================
# CHECK 2: Number of runs and positive rates
# ==========================================

overview["positive_rate"] = (
    overview["positive_0.5"] / overview["n_sites"]
)

print("\n=== NUMBER OF RUNS & POSITIVE RATES ===")
print(
    overview.groupby("cell_line")
    .agg(
        n_runs=("run", "count"),
        mean_sites=("n_sites", "mean"),
        mean_positive_rate=("positive_rate", "mean"),
        sd_positive_rate=("positive_rate", "std")
    )
    .to_string()
)

# ==========================================
# CHECK 3: Cross-cell-line similarity ranking
# ==========================================

print("\n=== CROSS-CELL SIMILARITY RANKING ===")
print(
    cross.sort_values("pearson", ascending=False)
    [["cell_a", "cell_b", "n_shared",
      "pearson", "spearman", "positive_jaccard_0.5"]]
    .to_string(index=False)
)

import pandas as pd

cross = pd.read_csv("outputs/tables/cross_cell_pairwise.csv")
rep = pd.read_csv("outputs/tables/replicate_pairwise.csv")

print("\n=== MCF7 REPRODUCIBILITY ===")
print(
    rep[rep.cell_line == "MCF7"]
    [["run_a", "run_b", "n_shared",
      "fraction_a", "fraction_b",
      "pearson", "spearman", "jaccard_0.5"]]
    .to_string(index=False)
)

print("\n=== CROSS-CELL PEARSON RANKING ===")
print(
    cross.sort_values("pearson", ascending=False)
    [["cell_a", "cell_b", "n_shared", "pearson"]]
    .to_string(index=False)
)

print("\n=== POSITIVE JACCARD RANKING ===")
print(
    cross.sort_values("positive_jaccard_0.5", ascending=False)
    [["cell_a", "cell_b", "positive_jaccard_0.5"]]
    .to_string(index=False)
)

print("\n=== WITHIN-CELL REPRODUCIBILITY ===")
print(
    rep.groupby("cell_line")
    .agg(
        n_pairs=("pearson", "count"),
        mean_pearson=("pearson", "mean"),
        mean_jaccard=("jaccard_0.5", "mean"),
        min_pearson=("pearson", "min"),
        max_pearson=("pearson", "max")
    )
    .round(4)
    .to_string()
)
