# Task 2: SG-NEx prediction explorer

Explore frozen prediction results for **23 nanopore direct RNA-seq runs from seven cell lines**. The Streamlit dashboard has five pages: Overview, Reproducibility, Cross-cell Similarity, Differential Atlas and Transcript Explorer. No model training or inference is needed to run it.

For online access, use the [deployed dashboard](https://dsa4262genomix.streamlit.app/).

## Run the dashboard

From the team repository root:

```sh
cd Task2
conda env create -f environment-task2.yml
conda activate task2-release
python -m streamlit run dashboard.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Alternatively, from the repository root, use a Python 3.12.9 virtual environment instead of Conda:

```sh
cd Task2
python3.12 -m venv .venv-task2
source .venv-task2/bin/activate
python -m pip install -r requirements-task2-lock.txt
python -m streamlit run dashboard.py
```

The tested environment uses **Python 3.12.9 and PyArrow 26.0.0**. Use the pinned dependencies for compatibility with the supplied Parquet files. The `outputs/` data are included; raw prediction CSVs are unnecessary for the dashboard. Restart the app or clear its cache after replacing data files.

The dashboard resolves data relative to `dashboard.py` and requires `dashboard.css` beside it. Under `outputs/tables/`, startup requires `prediction_overview.csv`, `replicate_pairwise.csv`, `cross_cell_pairwise.csv`, and `differential_atlas/atlas_manifest.json`. Each `differential_atlas/positive_{0.3,0.5,0.7,0.9}/` folder needs `differential_pair_summary.csv` and `differential_site_atlas.parquet` to support its cutoff. Keep all seven `consensus_<cell>.parquet` files for complete Transcript Explorer coverage. The detection-coverage table and fixed Figure 4 are optional; missing files omit their respective displays without preventing startup.

## Files and results

- `dashboard.css`: scoped typography, chart panels and responsive spacing; keep it beside `dashboard.py` when deploying.
- `dashboard.py`: interactive explorer; positive thresholds 0.3/0.5/0.7/0.9 and absolute score-difference thresholds 0.30–0.90 in steps of 0.05.
- `task2_pipeline.py`: generates summaries, consensus, pairwise comparisons and figures from existing prediction CSVs.
- `run_differential.py`: builds all four differential atlases from saved consensus files.
- `run_extended_analysis.py`: site conservation and between-/within-cell variability; writes `outputs/tables/extended_analysis/` and Figures 5–6.
- `assemble_report_figures.py`: creates the report composites; see [figure documentation](outputs/figures/report_figures_README.md).
- `check_task2.py`: prints run counts and within-/cross-cell comparison summaries.
- `check_mcf7.py`: optional diagnostic for exactly two raw CSVs at `sg_nex_prediction_results/MCF7/`, relative to the working directory; prints unnormalized transcript/site overlap and has no input-path argument.
- `outputs/tables/`: final scientific tables, seven consensus Parquet files and cross-cell score matrix. `differential_atlas/positive_0.x/` folders are the authoritative differential outputs; `atlas_manifest.json` documents their fields.
- `outputs/figures/`: Figures 1–6, supplementary Figures S1–S3, coverage and report composite figures, in PNG/PDF/SVG formats.
- `requirements-task2.txt`: direct dependencies; `requirements-task2-lock.txt`: pinned tested runtime dependencies; `environment-task2.yml`: Conda setup.
- `requirements.txt`: Streamlit Community Cloud entry point that includes `requirements-task2.txt`.

At positive threshold 0.5 and absolute mean-score difference ≥0.30, the atlas contains **48,355 selected pair-site records**, **38,909 strict records**, **15,272 distinct transcript-position sites** and **8,311 distinct transcript IDs**. All exported candidates have `selected=True`; `n_positive_runs_at_0_5_<cell>` always counts per-run positives at 0.5, regardless of folder.

## Optional analysis commands

Run from `Task2/` in the installed environment. Inspect the included results without rewriting them:

```sh
python check_task2.py
python -m unittest -v test_quality_control test_extended_analysis
```

Optional regeneration writes to `outputs/` and replaces the corresponding derived files. To rebuild from the original 23 prediction CSVs, use this order:

```sh
python task2_pipeline.py --input "/path/to/external/sg_nex_prediction_results" --output outputs
python run_differential.py --output outputs --min-runs 2
python run_extended_analysis.py --output outputs --input-dir "/path/to/external/sg_nex_prediction_results"
python assemble_report_figures.py --output outputs
```

Raw inputs must use `<input>/<cell>/<cell>_repN_runM.csv` paths and contain `transcript_id`, `transcript_position` and `score` columns.

If only rebuilding the atlases, start with `run_differential.py`; the included consensus files suffice. Extended analysis can run without `--input-dir` when `TASK2_RAW_DIR` is unset, but then omits individual run-pair distributions. Keep raw inputs outside the repository. Regeneration is unnecessary to use the included results or dashboard.

## Interpretation

Sites are normalized transcript-ID/position pairs, not deduplicated genomic loci. Cross-cell comparisons require detection in ≥2 runs in each cell. Pearson uses continuous scores on shared observations; Jaccard is both positive / either positive within those shared observations. Differential candidates additionally require the higher consensus mean to meet the chosen positive threshold; atlas selection uses the float32 tolerance documented below. Strict support means the higher cell's minimum observed run score exceeds the lower cell's maximum; it is descriptive, not statistical significance. Run pairs are dependent and are not independent biological replicates.

Ensembl version stripping assumes coordinate compatibility, particularly for MCF7, without remapping coordinates. Scores and differences are predictions, not experimentally validated methylation or calibrated methylation abundance.

## Prediction quality control and differential robustness

Run the QC assessment without regenerating the prediction pipeline or existing figures:

```sh
python quality_control.py --input-dir "/path/to/external/sg_nex_prediction_results" --output outputs --run-level-only
# Optional: omit --run-level-only to also regenerate differential QC.
python check_task2.py --qc-tests
```

`quality_control.py` reuses canonical normalized transcript-ID/position keys, cached run statistics, pairwise comparisons, detection coverage, site sharing, and the Hct116/K562 consensus Parquet files. It writes the following QC artifacts without replacing the original scientific tables or Figures 1–6/S1:

| Artifact | Contents |
| --- | --- |
| `outputs/tables/qc_run_summary.csv` | Per-run input metrics, provenance, audit availability, warnings |
| `outputs/tables/qc_overview.csv` | Compact totals, availability, reproducibility range, baseline regression checks |
| `outputs/tables/qc_heya8_shared_sites.csv` | Real-data shared-site HEYA8 comparison and cached-pair regression |
| `outputs/tables/qc_heya8_score_distributions.csv` | All/shared/run-only HEYA8 score summaries and exact quantiles |
| `outputs/tables/qc_reproducibility_summary.csv` | Per-run partner agreement, overlap, cell detection coverage, descriptive warnings |
| `outputs/tables/qc_differential_robustness.csv` | Fixed-universe threshold grid, strict support, transcript representation, baseline overlap and ranking stability |
| `outputs/figures/figS3_run_level_qc.{png,pdf,svg}` | Validated baseline positive rates and measured score distributions |
| `outputs/figures/figS2_qc_robustness.{png,pdf,svg}` | Local candidate-count, baseline-Jaccard and strict-support heatmaps; magenta outline marks the fixed baseline |

The dashboard presents scientific results through five pages, with brief reading guides and findings computed from the displayed results. Overview has a simple five-column run-statistics table and one short QC completion note. Detailed QC tables, diagnostics, downloads and Figures S2/S3 are retained here for reproducibility and reporting; they are not loaded or displayed by the dashboard. The Differential Atlas page retains scientific Figure 4 as a clearly labeled fixed Hct116/K562 baseline in a collapsed worked example, followed by threshold-dependent interactive comparisons.

### Run-level Quality Control

Run-level QC uses the fixed score threshold **0.5** for its internal assessment. It is documented below and preserved in the QC CSVs and figures. The visualization dashboard instead shows only a simple **Run-level Statistics** table: cell line, run, predicted sites, positive predictions and positive rate at the current interactive positive threshold. Changing the differential threshold does not affect that table. Differential threshold robustness is complementary to run-level QC, not a substitute for it.

Reproduce the run-level outputs without touching differential QC or Figures 1–4/S1/S2:

```sh
python quality_control.py --input-dir "/path/to/external/sg_nex_prediction_results" --output outputs --run-level-only
# Alternatively set TASK2_RAW_DIR externally; the script never persists its value.
# --input remains an alias; --chunksize defaults to 100000 records.
python check_task2.py --qc-tests
```

The completed real-data audit covers **23/23 runs and 28,670,080 raw records**. Every run has 100% valid records: zero missing IDs/positions/scores, malformed positions/scores, duplicate canonical keys (including across chunks), NaN/infinite scores or scores outside [0,1]. In these particular files, raw records, unique keys, valid scored sites and records meeting loader constraints are equal; the workflow computes each separately rather than assuming equality. Site counts, positive counts and score mean/median/sample SD agree with all 23 cached summaries; the observed moment differences are zero. The two HEYA8 descriptive warnings remain; neither is a structural validation failure.

The external source roster is checked against `prediction_overview.csv` before writing outputs: **23 runs across seven cell lines**, with the expected 2/6/5/3/2/3/2 run counts. Unrelated files are ignored by the existing cell/run filename convention. Inputs must remain outside the Git repository. Source metadata stores only relative `cell/run.csv` identifiers; raw input locations and parser exception paths are never exported.

Raw mode parses one file at a time in 100,000-row string chunks, converting positions to numeric values and scores to the pipeline's float32 representation. A temporary SQLite index counts canonical normalized transcript/position keys **across chunk boundaries**; disk storage scales with one run, and the database cache is capped at 16 MiB. Complete valid keys with finite probability scores and no duplicate membership define the diagnostic scored population. All members of duplicate keys are excluded from that diagnostic population, without choosing an arbitrary score or modifying scientific outputs. Both excess duplicates and total duplicate-member counts are reported. The current clean files satisfy the unchanged loader's all-or-reject constraints; records meeting those constraints are reported separately from raw, unique and valid record counts. No extra columns are required and no genomic coordinate remapping is performed.

Exact score statistics use one compact float32 vector per run (4 bytes per valid site, about 6.4 MB at 1.6 million sites), retained in original row order. Quantiles use exact linear interpolation, never a histogram/sample approximation; mean and sample SD use the same float32 population as the pipeline. Temporary SQLite data live in the operating system's temporary directory and are removed automatically. Only the HEYA8 investigation holds its two validated runs together for a canonical-key inner join. The cached comparison checks run names and site/positive counts exactly, and score mean/median/SD at `rtol=1e-5, atol=1e-7` to accommodate float32 arithmetic; discrepancies are recorded instead of forced to match. `n_cached_retained_records` remains separate from actual raw counts.

Warning rules are descriptive, **within each cell line**. With at least three measured runs, lower/upper Tukey fences (`Q1 − 1.5 IQR`, `Q3 + 1.5 IQR`) examine log2 site counts, positive-rate percentages, and mean scores; lower fences examine each run's mean shared-site Pearson and mean directional site-overlap fraction. Fences are stored in the CSV (count fences are in log2 units). Means describe score-distribution location, not its full shape. Absolute rates are not required to agree across cell lines. Small samples and shared partners make these fences exploratory, not statistical tests. With two runs, the table instead reports the site-count ratio and positive-rate span and explicitly notes that outlier attribution is ambiguous. A low positive rate by itself is not a failed run. Lowest-rank and repeated-replicate-label notes are retained as context, separate from anomaly counts.

**Two runs receive descriptive anomaly warnings:**

- `HEYA8_rep1_run2`: low site count (834,303), positive rate (0.434%), mean score (0.02423) and mean shared-site Pearson (0.5904), relative to HEYA8's stated fences. This confirms and retains the earlier warning.
- `HEYA8_rep1_run1`: high positive rate (0.927%) relative to the HEYA8 upper fence. This requires inspection, not a claim of sequencing failure or invalid prediction scores.

No directional-overlap lower-fence outlier was detected. `Review recommended` indicates measured anomalies or an observed structural file/parsing problem; `Input QC unavailable` indicates missing/incomplete raw audit coverage. Structural rejection is explicitly flagged even when a malformed file prevents calculating the full audit, so it is not mistaken for a clean file or merely an absent source. `No issues detected` is reserved for complete input audits without measured warnings. The summary reports completed raw validation separately from the **2 descriptive anomaly warnings**; structural validity does not imply biological or sequencing validation. Repeated runs may not be independent biological replicates. **No runs were excluded and no scores or biological results were changed.**

Figure S3 now contains two panels based on actual validated predictions: baseline positive rates for all 23 runs, and measured per-run score distributions. Boxes span the 25th–75th percentiles, median lines show the 50th percentile, and whiskers span the 5th–95th percentiles; tails outside those whiskers are omitted deliberately, not inferred. Each run uses the same percentile definition, independently of its site count. All computed 1st/5th/25th/75th/95th/99th percentiles remain available in the CSV.

**HEYA8 shared-site investigation.** The two rep1 runs share **735,268 sites**, covering **54.9% of run1** and **88.1% of run2** (site-set Jaccard 0.511). Shared-site Pearson is **0.5979** and Spearman **0.5923**, agreeing with the cached pairwise analysis. On this identical universe, mean scores are **0.04146 vs 0.02542**, median scores **0.01069 vs 0.00679**, 95th percentiles **0.19536 vs 0.10764**, and positive rates **1.322% vs 0.478%** for run1/run2. Thus the shared-site positive-rate gap (0.844 percentage points) exceeds the all-site gap (0.492 points). Run1 has 604,861 additional sites with a 0.447% positive rate; run2 has 99,035 additional sites with a 0.111% rate. Coverage composition dilutes the shared-site difference rather than accounting for it: score shifts remain after matching site identity. This does not identify a causal technical defect, prove biological variation, or justify calling either run failed. The two small diagnostic CSVs preserve all/shared/run-only score quantiles and pairwise results; no raw matched-site export is retained.

### Input metrics and deployment

The QC table contains actual raw counts, unique site counts, valid/invalid counts and percentages, missing-field counts, excess duplicates, duplicate-member counts, NaN/non-finite/infinite and out-of-range score counts, baseline positive counts/rates, mean/median/sample SD and exact quantiles. `nonfinite_scores` includes missing/malformed numeric scores; `nan_scores` and `infinite_scores` distinguish those cases. Diagnostic counts can overlap. Baseline `positive_rate = n_positive / n_valid_scored_sites`, with score ≥0.5 inclusive. Empty denominators are undefined (NA).

**Streamlit uses the scientific analysis tables and `dashboard.css`; Figure 4 is optional.** It does not require raw inputs or QC artifacts. Dashboard startup does not import the QC workflow, read the raw-input environment variable, discover source directories, parse raw CSVs or run audits. Removing QC CSVs or Figures S2/S3 has no effect on the scientific pages; the one QC note records the completed internal validation. The entry point alone reads `--input-dir` or `TASK2_RAW_DIR`. A source-roster mismatch stops the local audit before replacement outputs are written. Keep raw directories, local environment files and temporary databases out of Git; `.gitignore` preserves derived QC CSVs while excluding source run files. No new dependencies are needed (SQLite is in Python's standard library).

Prediction-level QC is not upstream sequencing/alignment QC. Read quality, alignment rates, mapping quality, RNA integrity, per-site read depth and signal errors are absent from the prediction CSV schema and are not inferred. Replicate independence remains unverified, and descriptive QC warnings do not establish technical failure or justify automatic exclusion.

### Reproducibility warnings and limitations

Prediction reproducibility indicates how consistent scores and positive calls are across observed runs, and helps identify candidates driven by individual runs. The QC reuses Pearson/Spearman scores and positive-site Jaccard at 0.5 **within shared observations**, plus directional site overlap and fractions of the cell's site union detected in ≥2 or all runs. Thus agreement is conditional on jointly observed sites; it does not account for all undetected sites. Cross-cell site-sharing totals are also summarized: detection across cell lines is not modification specificity.

Cell mean run-pair Pearson ranges from **0.590 to 0.670**. For cells with more than two runs, the lowest run mean is labeled descriptively, and means below the within-cell run-mean first quartile minus 1.5 IQR receive an additional warning. This is an explicit descriptive convention, not a validated exclusion threshold or a PASS/FAIL decision. **HEYA8_rep1_run2** is flagged by this rule (mean Pearson ≈0.590); no runs or candidates are removed. Small run counts and shared partners limit this comparison. A549, HepG2 and MCF7 each have two runs: a single pair cannot identify which run is responsible for disagreement or estimate between-pair variability.

Repeated `_repN` labels in H9, HEYA8 and Hct116 identify potentially repeated technical runs; different labels do not prove biological independence. The QC counts same-label and different-label comparisons and retains this metadata uncertainty. All 23 runs must not be treated as independent biological replicates. No hypothesis tests, p-values or FDR are computed.

### Differential sensitivity and interpretation

The Hct116–K562 matched universe is built once using the existing pipeline: **810,786 unique sites**, each observed in ≥2 runs per cell. Selection reuses the atlas rule: `abs_delta_mean >= delta_threshold - 1e-7` and `higher_mean_score >= positive_threshold - 1e-7`. The tolerance is the existing float32 boundary convention from `run_differential.py`. The earlier focus CSVs use exact comparisons; both conventions produce the same baseline here: **2,736 selected candidates and 2,230 strict candidates (81.5%)**. Regression checks compare canonical site sets against both saved focus CSVs and the authoritative atlas. Existing results and Figures 1–4/S1 are not overwritten.

Figure S2 uses a **local grid of 25 combinations**: higher-cell mean-score thresholds 0.40/0.45/0.50/0.55/0.60 crossed with absolute mean-score difference thresholds 0.20/0.25/0.30/0.35/0.40. All comparisons use the same 810,786 matched sites, including combinations below the atlas export cutoff; selected-only atlases are therefore not used as the evaluable universe. The CSV retains the earlier broad grid (positive 0.3/0.5/0.7/0.9; difference 0.3/0.4/0.5/0.6/0.7), with **43 unique combinations** in total. The semicolon-delimited `grid_id` is `local`, `broad`, or `local;broad` for the two shared combinations. Each shared combination is stored once. Jaccard is candidate intersection divided by union against the fixed baseline; baseline retention is intersection divided by baseline size. Thus a relaxed selection can retain 100% of baseline candidates while its Jaccard falls because additional candidates enter. `strict_fraction` is strict count divided by selected count; `baseline_retention` is a proportion, and the existing percentage columns are retained for compatibility. For zero selected candidates, strict fraction is undefined (NA), not zero. All other empty denominators are also NA. Transcript stability compares the top 20 transcripts ranked by strict-supported site count, selected-site count, then maximum absolute score difference (all descending), with transcript ID ascending as the deterministic final tie break. This extends the existing ranking only for exact ties. The CSV gives top-set Jaccard, baseline top-20 retention, ordered IDs and mean absolute rank movement among common top transcripts; the last measure excludes dropped transcripts and should be read with retention.

**Report-ready local assessment.** Across the fixed 810,786-site Hct116/K562 universe, the baseline (positive ≥0.50, absolute difference ≥0.30) selects 2,736 candidates, including 2,230 with strict observed-run range separation (81.5%). Holding positive threshold at 0.50, changing the difference cutoff to 0.25 selects 3,522 sites (Jaccard 0.777, baseline retention 100%, strict support 74.0%); increasing it to 0.35 selects 2,016 (Jaccard 0.737, retention 73.7%, strict support 88.1%). Holding difference cutoff at 0.30, positive cutoffs of 0.45 and 0.55 select 3,611 and 1,968 sites, respectively (Jaccard 0.758/0.719; retention 100%/71.9%; strict support 80.6%/83.0%). Across the entire local grid, selected counts range from 795 to 7,925, Jaccard from 0.291 to 1.000, baseline retention from 29.1% to 100%, and strict support from 62.4% to 95.2%. Single-axis 0.05 changes preserve a substantial majority of baseline membership, but the membership changes are material: the baseline region is locally sensitive rather than invariant. These observations describe prediction-set sensitivity, not biological robustness.

Candidate counts cannot increase as either threshold increases; that nesting alone is not evidence of unreliability. Higher difference cutoffs enrich for strict-supported sites in this local grid: at positive 0.50, strict support rises from 65.5% at difference 0.20 to 94.5% at 0.40. This enrichment is partly mechanical: both the selection difference and the separation criterion derive from the same observed scores. It supplies no independent validation and does not establish the baseline thresholds as optimal. Positive-threshold tightening also enriches strict support here, but that empirical pattern is not guaranteed for other datasets or parameter ranges.

Strict-support percentages in the local grid (rows: higher-cell positive threshold; columns: absolute difference threshold):

| Positive threshold | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 62.4% | 71.7% | 80.4% | 88.5% | 93.9% |
| 0.45 | 63.7% | 72.1% | 80.6% | 88.4% | 94.2% |
| 0.50 | 65.5% | 74.0% | 81.5% | 88.1% | 94.5% |
| 0.55 | 68.9% | 76.4% | 83.0% | 89.2% | 94.8% |
| 0.60 | 71.6% | 78.5% | 84.6% | 90.1% | 95.2% |

Strict observed-run range separation means the higher-mean cell's **minimum** observed run score exceeds the lower-mean cell's **maximum**, i.e. `conservative_gap > 0` (touching ranges do not qualify). This criterion remains fixed across the threshold grid and uses only detected runs. It is descriptive consistency, not statistical significance, an uncertainty model, or evidence from independent biological replication. Score differences can reflect model behavior, coverage and run effects; they do not establish experimentally validated differential m6A modifications or calibrated methylation changes. Transcript-version coordinate compatibility remains assumed. Biological interpretation requires independent experimental evidence and a defensible replicate design.

Run `python -m unittest -v test_quality_control test_extended_analysis` from `Task2/` to verify QC, threshold boundaries, candidate-set regression, site sharing and variability. Streamlit AppTest coverage includes all five pages, threshold persistence across navigation, transcript controls, and operation with absent or partial QC artifacts. `python check_task2.py --qc-tests` runs only the QC test module. These checks use saved outputs and temporary fixtures; they do not rerun the full external raw-data audit. Streamlit Community Cloud uses `requirements.txt`, which includes the pinned direct dependencies.

## Dashboard navigation

Each page addresses a separate question: coverage and positive rates (Overview), within-cell agreement and cutoff sensitivity (Reproducibility), between-cell similarity (Cross-cell Similarity), candidate prioritization (Differential Atlas), and site positions with optional observed run ranges (Transcript Explorer). Overview shows coverage as run-level dots with median markers and zero horizontal jitter. Reproducibility retains the Jaccard threshold-sensitivity curve and run-pair table; the standalone Jaccard bar chart has been removed.

The cross-cell matrix switches between continuous-score correlation and positive-site overlap. The fixed Hct116–K562 figure is available as a collapsed worked example. Ranking details are also collapsed to avoid repeating the plotted values. The positive cutoff defaults to **0.5**, with options **0.3/0.5/0.7/0.9**. The score-difference cutoff defaults to **0.30**, ranges from **0.30 to 0.90** in **0.05** steps, and appears only on Differential Atlas and Transcript Explorer. Both cutoffs persist across page navigation. In Transcript Explorer, choose the comparison pair and displayed cell lines, then optionally show observed minimum–maximum ranges on the same score plot. Ranges are not confidence intervals. Transcript rankings are counts rather than enrichment estimates.
