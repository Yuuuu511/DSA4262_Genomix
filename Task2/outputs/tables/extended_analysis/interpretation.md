# Extended Task 2 analytical notes

These are analysis findings and integration recommendations, not a rewrite of the team report. They describe model predictions, not experimentally established methylation or statistical significance.

## Reproduce and interpret the outputs

```sh
python run_extended_analysis.py --output outputs --input-dir /path/to/external/sg_nex_prediction_results
# Alternatively provide TASK2_RAW_DIR externally.
python -m unittest -v test_extended_analysis
```

No new dependencies are required. Cached Parquet tables provide all consensus, detection-count, range and SD information. Only the six Hct116/K562 source files are read, individually, for individual-run pair comparisons; only their aligned float32 score arrays remain in memory. No source paths or raw CSV copies are exported. The optional raw-free execution still reports cached means/ranges/SDs; individual-run pair distributions are then explicitly unavailable.

A site is the existing normalized transcript ID/position pair. ENST version suffixes are removed by the unchanged loader, with transcript-coordinate compatibility assumed rather than remapped. These are transcript sites, not deduplicated genomic loci. Cell-line scores are means over observed valid runs; an undetected site is not a negative prediction. Positive calls use exact score >= threshold, as in the existing cross-cell analysis. The primary universe requires scores from at least two runs in every one of the seven cell lines. Differential selection reuses `candidate_mask` and its existing 1e-7 float32 tolerance; exact legacy and atlas baseline candidate sets are verified to agree.

The CSV headers state their denominators. `a_sharing_distribution.csv` includes the all-negative category separately; its percentage of the positive union is undefined. Cell-specific fractions divide by that cell's positive calls in the same complete universe, not all its scored sites. `b_ecdf.csv` contains exact empirical fractions at 0.001-spaced score differences, not sampled observations; the plot connects these grid evaluations. Numeric summaries and quantiles are exact. Figure 6A zooms the horizontal axis to 0–0.10 without changing its denominator or excluding high-difference sites.

## A. A partly shared, threshold-dependent positive landscape

**Evidence and interpretation.** The primary seven-cell universe contains **440,569 sites**, or **19.30% of the 2,282,424 sites observed in at least two runs in any cell line**. At score >=0.5, **430,826 sites are negative in all seven cells** and **9,743 (2.211% of eligible sites) are positive in at least one**. Among those 9,743 positive sites, **5,684 (58.34%) are shared by at least two cell lines**, **4,059 (41.66%) are positive in exactly one**, and **527 (5.41%) are positive in all seven**. Thus a majority of the observed positive union is shared, but universal positivity is uncommon. The evidence supports partial sharing with a substantial cell-restricted predicted component, rather than a uniformly conserved positive landscape.

| Positive cell lines | Sites | Percentage of positive union (9,743) |
| ---: | ---: | ---: |
| 1 | 4,059 | 41.66% |
| 2 | 1,978 | 20.30% |
| 3 | 1,165 | 11.96% |
| 4 | 845 | 8.67% |
| 5 | 659 | 6.76% |
| 6 | 510 | 5.23% |
| 7 | 527 | 5.41% |

The all-seven and exactly-one categories represent **0.1196% and 0.9213% of all eligible sites**, respectively. These percentages have a different denominator from the positive-union percentages above.

**What the restricted calls mean.** The counts below identify a positive consensus in one cell and negative consensuses in the other six; they do not establish a cell-specific molecular modification. Among all 4,059 calls, only **1,349 (33.23%)** are positive in every observed run of the positive cell. Median positive-cell consensus scores are **0.531–0.561**, close to the 0.5 cutoff. These calls therefore include substantial run-level disagreement and borderline classification.

| Only positive cell line | Specific sites | Positive sites in this cell, common universe | Specific / this cell's positives | Specific calls positive in all observed runs | Median observed runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| A549 | 474 | 3,300 | 14.36% | 178 (37.55%) | 2 |
| H9 | 110 | 2,031 | 5.42% | 2 (1.82%) | 6 |
| HEYA8 | 197 | 2,689 | 7.33% | 18 (9.14%) | 5 |
| Hct116 | 1,058 | 5,331 | 19.85% | 222 (20.98%) | 3 |
| HepG2 | 1,158 | 5,136 | 22.55% | 545 (47.06%) | 2 |
| K562 | 414 | 2,749 | 15.06% | 111 (26.81%) | 3 |
| MCF7 | 648 | 3,698 | 17.52% | 273 (42.13%) | 2 |

**Coverage limitation.** The shared universe is large enough for a meaningful conditional comparison, but excludes 80.70% of the any-cell evaluable union. It retains only **30.13–50.24%** of each cell's evaluable sites, while retaining **56.23–71.32%** of its positive sites. This is evidence of ascertainment toward well-observed, positive-enriched sites. It does not identify whether expression, sequencing depth, model behavior or other factors cause that enrichment. No absent site is counted as negative, and no secondary comparison silently expands the denominator. The results cannot be generalized to all transcript positions, all genes, or all biological m6A sites.

**Sensitivity limitation.** On the same 440,569 sites, thresholds 0.45/0.50/0.55 give 12,547/9,743/7,368 positive-union sites. The multi-cell fractions are **61.92% / 58.34% / 54.33%**; all-seven fractions are **6.73% / 5.41% / 4.04%** of their corresponding positive unions. The qualitative finding of majority multi-cell sharing persists. However, only **61.12%** of baseline specific calls remain specific to the same cell at 0.45, and **51.24%** at 0.55. Specific-set membership is not nested under threshold changes, because additional cells may enter or leave the positive set. These results do not establish robust biological cell specificity.

## B. Large selected differences often exceed run variation, but selection does not imply low variability

**Eligibility and weighting.** All **810,786** original Hct116/K562 matched sites have at least two observed scores per cell; **zero sites are removed**. The run-count combinations Hct116/K562 are 2/2: **75,260**, 2/3: **59,224**, 3/2: **216,701**, and 3/3: **459,601**. Cached means, minima, maxima, sample SDs and counts were checked against the six raw run files. For run-pair comparisons, each site contributes the mean absolute difference across its observed within-cell unordered pairs or cross-cell Cartesian pairs. The resulting distributions give each site one observation; sites with nine cross-run combinations do not receive more weight than sites with four. This prevents pair-level pseudo-replication, but does not make sites or biological samples independent.

**Background versus candidate differences.** Across all matched sites, median consensus difference D is **0.00678**, compared with median within-Hct116 and within-K562 mean pair differences **0.00985 and 0.00697**. The median cross-cell individual-run pair difference is **0.01112**. Averaging run scores reduces differences and can cancel opposing run effects; consensus and individual-run differences must not be conflated. Ordinary matched sites do not show a broad shift that clearly exceeds observed run variation.

For the **2,736** baseline candidates, median D is **0.4040**; median within-Hct116 and within-K562 mean pair differences are **0.1612 and 0.1449**. At the site level, **2,347/2,736 (85.78%)** have D greater than both within-cell mean pair differences, and **1,760/2,736 (64.33%)** have D greater than the larger observed within-cell range. These are descriptive magnitude comparisons; selecting D >=0.3 already favors such outcomes and is not independent validation.

**What strict separation adds.** The original **2,230/2,736 (81.51%)** strict-supported candidate sites are recovered exactly, including their identities. Their higher-cell minimum exceeds their lower-cell maximum. The remaining **506** selected sites have overlapping or touching observed ranges. Among candidates, median maximum within-cell range is **0.2883 for strict-supported sites versus 0.5297 for the others**. This makes strict support useful for prioritizing within the selected set. It does not mean a confidence interval was separated or a statistical test was passed.

The D > maximum-range comparison is not equivalent to strict separation: **484 strict-supported candidates do not satisfy D > maximum range**, while **14 range-overlapping candidates do**. Means can lie asymmetrically within three-run ranges. Both summaries therefore remain distinct and neither replaces the canonical strict criterion.

**A qualification to reliability claims.** Selected candidates are **not generally less variable than other matched sites**. Their median maximum within-cell range is **0.3259**, versus **0.02091** in the other 808,050 sites (about **15.6-fold larger**); Hct116/K562 sample SD medians are **0.1260/0.1124** in candidates versus **0.00764/0.00528** elsewhere. Scores are bounded, and high-score selection and low-score background distributions affect these contrasts. No score-matched noise or biological replicate model is available to separate those effects. Large between-cell differences can coexist with substantial within-cell variation.

**Run coverage and threshold checks.** Among the **1,613 candidates observed in all three runs of both cells**, **1,235 (76.57%)** retain strict separation; the qualitative result persists, but the percentage is lower than the 81.51% pooled value. More observed runs can reveal a wider range. Raising the difference cutoff from 0.30 to 0.40 retains 1,414 candidates, 1,336 strict-supported (**94.48%**); at 0.50, 526 remain, 521 strict-supported (**99.05%**). Selection and separation use the same scores, so this enrichment is partly mechanical. Falling counts do not prove robustness and these checks do not identify an optimal threshold.

**Are a few extreme sites responsible?** **372 candidates (13.60%)** fall in the top 1% of maximum ranges across the entire matched universe, so candidates are enriched for highly variable sites. However, the **28 highest-range candidate sites** contribute only **2.42%** of the sum of candidate maximum ranges. Together with the high candidate median, this shows that variability is not confined to a handful of extreme sites. This is a concentration diagnostic, not a variance decomposition or a statistical outlier test; no sites were removed.

**Remaining uncertainty.** Hct116 includes repeated replicate identifiers and the runs cannot all be assumed biologically independent. Observed ranges depend on how many runs detect the site; unobserved runs have no score. Model calibration, coverage mechanisms, transcript-version coordinate compatibility, correlations among transcript sites/isoforms, and biological confounding remain unresolved. There are no p-values, FDR estimates, formal null distributions or claims of experimental verification.

## Recommendation for the five-page team report

Both analyses add meaningful interpretation. A directly distinguishes shared positive calls from shared detection; B qualifies what makes a differential candidate worth prioritizing. B provides the stronger addition to the differential narrative because it shows both the value of strict separation and the failure of an unconditional “selected sites are more reproducible” claim.

Include **at most one compact two-panel new main figure**, combining the sharing distribution from **Figure 5A** with the difference-versus-variability plot from **Figure 6B**. This is a recommendation only; no third/composite figure or report rewrite has been created. Give it space by shortening the existing **Figure 3 cross-cell similarity heatmaps** to one sentence or moving them to supplementary material: score correlation and positive overlap have already been quantified, whereas the new panels answer more directly what is shared and whether selected differences exceed run variation. Avoid repeating Figure 4's density and score-difference histogram alongside Figure 6B; retain its baseline counts and, only if space permits, one transcript example. Keep Figure 5B's cell breakdown, Figure 6A's ECDF, per-run coverage, strict/gap contingency, specificity identity stability, and threshold diagnostics in reproducible outputs rather than the five-page main text.

The defensible narrative is **partially shared predictions with threshold-dependent cell-restricted calls, and a selected differential subset often separated across observed runs despite substantial absolute run variability**. Neither analysis establishes biological conservation, statistical significance, or validated differential methylation.
