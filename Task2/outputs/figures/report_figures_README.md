# Final report composite figures

These figures recompose five existing panels. They add no scientific analysis and do not require raw prediction CSVs. Existing figures, tables, QC artifacts and the Dashboard are unchanged.

## Files and final size

- `report_fig1_global_patterns.{pdf,svg,png}`: 178 × 82 mm, three horizontal panels.
- `report_fig2_differential_candidates.{pdf,svg,png}`: 178 × 81 mm, two horizontal panels.

Use PDF for LaTeX. PNG previews are 600 dpi. SVG retains editable text. PDFs embed Arial TrueType subsets; SVG viewers need Arial installed (or will substitute a font). Regeneration falls back to DejaVu Sans if Arial is unavailable.

At the native width, axis labels are 7.5 pt, ticks 7 pt, panel titles 8 pt, panel letters 9.5 pt, legends approximately 6.7 pt, and heatmap values 6.5 pt. N/A diagonal labels are 6 pt. Avoid reducing these figures to a single report column.

## Source panels and verified datasets

| Composite panel | Existing source | Inputs and preserved contents |
| --- | --- | --- |
| Figure 1A | `fig2_replicate_correlation` (Figure 2A) | `outputs/tables/replicate_pairwise.csv`: all 34 within-cell run-pair Pearson coefficients, computed on each pair's shared transcript-position sites. Original cell-line order/colors and absence of jitter retained. Coincident points can overlap. |
| Figure 1B | `fig3_cross_cell_similarity` (Figure 3A) | `outputs/tables/cross_cell_pairwise.csv`: all 21 pairwise Pearson values, mirrored to a seven-cell matrix. Each cell-line consensus requires at least two observed runs; correlations use the pair's shared eligible sites. Exact original Blues scale limits from `nice_limits` retained; values displayed to three decimals. Diagonal is N/A, not zero or a measured self-correlation. |
| Figure 1C | `fig5_site_conservation` panel A | `outputs/tables/extended_analysis/a_sharing_distribution.csv`, threshold 0.5: counts 4,059; 1,978; 1,165; 845; 659; 510; 527 for exactly 1–7 positive cell lines. Universe: 440,569 sites observed in at least two runs in each of all seven cell lines. Only the 9,743 sites positive in at least one cell are plotted; 430,826 all-negative sites are omitted. Positive means cell-line consensus mean ≥0.5. Missing sites were excluded from this common universe, never treated as negative. |
| Figure 2A | `fig6_between_vs_within_variability` panel B | `outputs/tables/consensus_Hct116.parquet` and `consensus_K562.parquet`, joined through the unchanged `task2_pipeline.build_differential_pair`. All 810,786 matched sites with ≥2 observed runs in each cell, the same 65-grid log-count gray hexagons, and all 2,736 selected site identities retained. Selection: higher-cell consensus mean ≥0.50 and absolute mean difference D ≥0.30. Magenta: 2,230 strict-supported sites; orange: 506 sites with overlapping/touching ranges. Strict support: the higher-mean cell's minimum observed run score exceeds the lower-mean cell's maximum (`conservative_gap > 0`). x is the maximum of the two observed within-cell ranges. D = maximum range is a descriptive reference, not the strict-support boundary. |
| Figure 2B | `fig4_differential_Hct116_vs_K562` panel D | Same cached consensus join, restricted to `ENST00000299238`: all 39 matched positions and both mean-score series. The same 15 selected positions (13 strict-supported) have hollow rings on each cell-line score. Hct116 remains blue `#4477AA`; K562 remains orange `#EE8866`. Dashed line at score 0.50. Scores are discrete points, not interpolated curves. |

The differential selection and strict site sets are checked against the existing `differential_selected_Hct116_vs_K562.csv` and `differential_strict_Hct116_vs_K562.csv`. The original plotting code is in `task2_pipeline.py` and `run_extended_analysis.py`; the canonical threshold helper also remains in `run_differential.py`. The assembly uses the exact original Figure 4 selection, with regression checks against saved identities; it does not invoke a figure function that writes analysis tables.

## Purely visual adjustments

- Fixed 178 mm white canvases without oversized figure-level titles or tight cropping.
- Arial fonts, thinner spines, compact external legends, smaller proportional markers, and consistent panel labels.
- Rotated cell-line tick labels to retain readable names at report width. Heatmap receives more width; its colorbar is horizontal below it. Gray N/A diagonal tiles are explicit.
- Sharing count labels retain thousands separators. Its title wraps to fit; the common universe and omitted all-negative category are stated below it.
- Figure 2A hexagons and candidate dots are rasterized at 600 dpi to keep vector files compact. Their input data, grid, log normalization, and classifications are unchanged. Axes, text, reference lines, legends, and the complete transcript panel remain vector objects. Figure 1 is vector throughout.
- Figure 2 reference-line meanings are explicit in its legend and LaTeX caption. The former explanatory footer is omitted, and 4 mm of empty bottom canvas is trimmed without resizing or moving plotted content. The diagonal does not classify strict support. Ranges are observed extrema, not confidence intervals.

## Regenerate

From the Task2 directory, using the existing project environment:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/task2-mpl python assemble_report_figures.py --output outputs
```

No raw-data argument, inference, downloads, or scientific-output regeneration is needed. The script saves only the six named report figure exports. It asserts baseline counts, candidate/strict identities, sharing counts, plotted pair-score values, transcript highlights, complete hexbin site counts, and rendered page bounds. Existing dependencies suffice.

## Recommended LaTeX captions and inclusion

Copy the two PDFs into the report's relative `figures/` directory. Use `figure*` across both columns; the source report need not be modified by this workflow. The captions below are report integration suggestions, not edits to the report.

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/report_fig1_global_patterns.pdf}
    \caption{Global patterns of predicted m6A scores. (A) Each dot is one
    within-cell run-pair Pearson correlation at shared transcript-position sites.
    (B) Pearson correlations between cell-line consensus scores at shared eligible
    sites (at least two observed runs per cell line); diagonal entries are not
    applicable. (C) Numbers of sites positive in exactly one to seven cell lines,
    using consensus scores $\geq 0.5$. The common universe contains 440,569 sites
    observed in at least two runs in every cell line. Only 9,743 sites positive in
    at least one cell are shown; 430,826 all-negative sites are omitted.
    Missing sites are never treated as negative.}
    \label{fig:global_patterns}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/report_fig2_differential_candidates.pdf}
    \caption{\textbf{Differential m6A prediction candidates between Hct116 and K562.}
    (A) Absolute difference in consensus mean prediction scores ($D$) versus the
    maximum observed within-cell run-score range across 810,786 matched sites.
    Gray hexagons show site density; magenta and orange points indicate selected
    candidates with strictly separated and overlapping observed run-score ranges,
    respectively. Candidates require $D \geq 0.30$ and a higher cell-line consensus
    mean score $\geq 0.50$. The horizontal dotted line marks $D = 0.30$, while the
    diagonal dashed line indicates $D$ equal to the maximum within-cell range;
    the diagonal is a descriptive reference, not the criterion for strict range
    separation. (B) Mean prediction scores along transcript ENST00000299238 for
    Hct116 and K562. Hollow circles indicate selected differential candidate sites,
    and the horizontal dashed line marks the positive-score threshold of 0.50.}
    \label{fig:differential_candidates}
\end{figure*}
```

## Verification

The assembly was checked against all five source plots and their saved data. PDF pages retain the stated physical dimensions and embedded Arial fonts. Both PNG exports and PDF raster previews were inspected, including a native-width preview. Matplotlib page-bound checks guard against clipped text/axes. Original scientific/QC outputs and existing source files were checked against a pre-assembly SHA-256 snapshot. No raw CSVs, local absolute paths, new dependencies, or changes to the Dashboard/report are introduced.
