# Task 2: SG-NEx prediction explorer

Explore frozen prediction results for **23 nanopore direct RNA-seq runs from seven cell lines**. The Streamlit dashboard has five pages: Overview, Reproducibility, Cross-cell Similarity, Differential Atlas and Transcript Explorer. No model training or inference is needed to run it.

## Run the dashboard

From the team repository root:

```sh
cd Task2
conda env create -f environment-task2.yml
conda activate task2-release
python -m streamlit run dashboard.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). For a Python 3.12.9 virtual environment instead of Conda:

```sh
python3.12 -m venv .venv-task2
source .venv-task2/bin/activate
python -m pip install -r requirements-task2-lock.txt
python -m streamlit run dashboard.py
```

The tested environment uses **Python 3.12.9 and PyArrow 26.0.0**. Keep the pinned dependencies: PyArrow 19.0.0 cannot read the supplied consensus Parquet files. The `outputs/` data are included; raw prediction CSVs are unnecessary for the dashboard. Restart the app or clear its cache after replacing data files.

## Files and results

- `dashboard.py`: interactive explorer; positive thresholds 0.3/0.5/0.7/0.9 and absolute score-difference thresholds 0.30–0.90 in steps of 0.05.
- `task2_pipeline.py`: generates summaries, consensus, pairwise comparisons and figures from existing prediction CSVs.
- `run_differential.py`: builds all four differential atlases from saved consensus files.
- `check_task2.py`: prints run counts and within-/cross-cell comparison summaries.
- `check_mcf7.py`: inspects optional raw MCF7 inputs; zero exact raw-ID overlap is expected before transcript-version normalization.
- `outputs/tables/`: final scientific tables, seven consensus Parquet files and cross-cell score matrix. `differential_atlas/positive_0.x/` folders are the authoritative differential outputs; `atlas_manifest.json` documents their fields.
- `outputs/figures/`: final Figures 1–4 and supplementary `figS1_site_sharing`, in PNG/PDF/SVG formats.
- `requirements-task2.txt`: direct dependencies; `requirements-task2-lock.txt`: pinned tested runtime dependencies; `environment-task2.yml`: Conda setup.

At positive threshold 0.5 and absolute mean-score difference ≥0.30, the atlas contains **48,355 selected pair-site records**, **38,909 strict records**, **15,272 distinct transcript-position sites** and **8,311 distinct transcript IDs**. All exported candidates have `selected=True`; `n_positive_runs_at_0_5_<cell>` always counts per-run positives at 0.5, regardless of folder.

## Optional analysis commands

Run from `task2/` in the installed environment:

```sh
python check_task2.py
python run_differential.py --output outputs --min-runs 2

# Requires the original 23 CSVs, retained locally and excluded from Git:
python task2_pipeline.py --input sg_nex_prediction_results --output outputs
python check_mcf7.py
```

The large full matched-site focus CSV, raw predictions, duplicate ZIP/per-pair exports, and superseded two-cell notebook/script are excluded from submission and preserved locally. Existing final figures and results are used unchanged; regeneration is optional.

## Interpretation

Sites are normalized transcript-ID/position pairs, not deduplicated genomic loci. Cross-cell comparisons require detection in ≥2 runs in each cell. Pearson uses continuous scores on shared observations; Jaccard is both positive / either positive within those shared observations. Differential candidates additionally require the higher consensus mean to exceed the chosen positive threshold. Strict support means the higher cell's minimum observed run score exceeds the lower cell's maximum; it is descriptive, not statistical significance. Run pairs are dependent and are not independent biological replicates.

Ensembl version stripping assumes coordinate compatibility, particularly for MCF7, without remapping coordinates. Scores and differences are predictions, not experimentally validated methylation or calibrated methylation abundance.
