# Genomix / haribo: m6A site prediction

This Task 1 project trains **one XGBoost classifier using data0 only**, and uses
that same saved model to predict dataset0, dataset1 and dataset2. Dataset1 and
dataset2 labels were never used for fitting, early stopping, or model selection.
The intermediate submission deadline is October 7, 2026, 23:59 Singapore time.
The slash in the team name is replaced by an underscore in filenames.

## Start in Visual Studio Code

Open this folder in VS Code, create a virtual environment, and select it with
**Python: Select Interpreter**.

Tested versions are pinned in `requirements.txt`. Python 3.10–3.13 is the intended
runtime range; this run was verified on Python 3.13/macOS ARM64. Ubuntu has not
been tested in this session. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Predict immediately using the included trained model

No labels or metadata are needed for prediction. Both newline-delimited `.json`
and `.json.gz` files are accepted. The output directory is created automatically.

```bash
python predict.py --data example/test.json.gz --model artifacts/model.joblib --output example/predictions.csv
```

`example/expected_predictions.csv` contains the expected output for the included
20-site smoke-test dataset. Scores should match within 1e-6 with pinned versions.
`example/labels.csv` is provided only for context; prediction never reads it.

## Retrain and generate all three submissions

```bash
python run_pipeline.py --data-root /path/to/Group_Project --team-name Genomix_haribo
```

Replace `--data-root` with the downloaded data folder on another computer. The
wrapper trains exclusively on `data0/dataset0.json.gz` and
`data0/data.info.labelled`, predicts all three datasets, then validates exact
coverage. Use `--skip-training` to reuse the included model. Feature caching
reduces repeat runs; cached files contain features, not labels.

For training alone:

```bash
python train.py --data /path/to/data0/dataset0.json.gz --labels /path/to/data0/data.info.labelled --output-dir artifacts --cache-dir .cache
```

For prediction on any new sample:

```bash
python predict.py --data /path/to/data.json.gz --model artifacts/model.joblib --output predictions.csv
```

## Method and evaluation protocol

The target is a **transcript-position site**, not an individual read. Each read
contains dwell time, signal standard deviation, and mean current for the −1,
central and +1 positions. There are 110 site features:

- Mean, standard deviation, 10th percentile, median and 90th percentile of each of the nine signals.
- Mean and standard deviation of log-transformed dwell times and central-minus-flanking signal differences.
- Log read count, seven-base sequence one-hot encodings and the 18 central DRACH motif encodings.

Transcript IDs, gene IDs, transcript positions, labels and metadata byte offsets
are **not input features**. Quantiles and spread retain some information about
heterogeneous reads, although aggregation loses read-level dependencies and does
not estimate which individual reads are modified.

Five reproducible stratified group folds, with `gene_id` as the grouping variable,
give approximately 60% training, 20% validation and 20% test sites. No gene crosses
these splits. Four XGBoost candidates vary depth (3/5) and positive-class weight
(1/4); early stopping uses validation PR AUC. The average of validation ROC AUC
and trapezoidal PR AUC selects the model. The test set is evaluated once after
selection, using a model refit on training + validation genes. Finally, one
submission model is refit on all data0 with the selected fixed boosting rounds.

The simple baseline is logistic regression using the nine mean signal features,
with scaling fitted on the training partition. Metrics include ROC AUC,
trapezoidal PR AUC, average precision (a different PR summary), and Brier score.
Positive-class reweighting candidates use an analytic prior correction; this
does not constitute empirical probability calibration. The selected model has
weight 1, so that correction is the identity.

See `RESULTS.md` and the JSON files in `artifacts/` for measured results and
limitations. Data0 leaderboard predictions are in-sample because the final model
uses all data0; their performance is not independent validation. Datasets 0 and
1 overlap by transcript-position and sometimes have different labels. The
external evaluation reports both all dataset1 sites and a subset restricted to
data0's heldout genes using the separate evaluation model. No external results
are used to change the submitted model.

## Submission checks

The generated submission folder contains these three CSVs. The existing
intermediate CSVs are stored separately at repository root in
`Intermediate_leaderboard_submission/`:

- `Genomix_haribo_dataset0.csv`: 121,838 sites.
- `Genomix_haribo_dataset1.csv`: 90,810 sites.
- `Genomix_haribo_dataset2.csv`: 1,323 sites.

Each has `transcript_id,transcript_position,score`, with finite scores in [0,1]
and exactly one prediction per JSON site. There is no index column. Dataset2
identifiers such as `tx_id_0` and position 0 are preserved. Its fractional labels
are not silently converted to binary labels.

```bash
python validate_submission.py --data-root /path/to/Group_Project --submission-dir submissions --team-name Genomix_haribo --report artifacts/submission_validation.json
```

The course allows no resubmission. Check the exact team spelling and upload
instructions on Canvas before uploading **only the three CSVs**. Nothing has
been uploaded or submitted by this project.

## m6Anet comparison

The benchmark uses official m6Anet 2.0.1 HCT116 network weights, normalization
factors and kmer mapping, via `benchmark_m6anet.py`. The course JSON has nine
signal columns rather than the read-ID column expected by the current CLI, so
the adapter reads those signals directly. It computes the exact expectation of
20-read, sampling-with-replacement noisy-OR pooling instead of the upstream
finite Monte Carlo approximation. An explicit simulation checks that pooling.
This is an **adapter benchmark**, not an unmodified official CLI run.

Pretrained m6Anet overlap with the course data is unknown. Its comparison on our
heldout genes must not be described as a fully independent comparison of training
procedures. Our model aggregates site statistics; m6Anet embeds sequences and
learns read probabilities with multiple instance learning.

The handout recommends a separate Python 3.8 conda environment for official
m6Anet. Keep that separate from the primary model environment. For reproducing
the exact modern-runtime adapter used here, install optional dependencies in an
isolated target (do not allow legacy dependency pins to replace the main stack):

```bash
python -m pip install --target .benchmark-deps --no-deps m6anet==2.0.1 toml==0.10.2 ujson==6.0.0
python -m pip install torch==2.6.0 tqdm==4.67.1 setuptools
python benchmark_m6anet.py --data /path/to/data0/dataset0.json.gz --heldout-predictions artifacts/heldout_predictions.csv --output-dir artifacts --vendor-path .benchmark-deps
```

This adapter was tested with modern dependencies and does not claim conformance
with upstream's legacy pinned environment. Official project and citation:
[GoekeLab/m6anet](https://github.com/GoekeLab/m6anet), Hendra et al. (2022),
[Nature Methods](https://doi.org/10.1038/s41592-022-01666-1).
XGBoost API: [official documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
Grouped splitting: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

## Final-project work remaining

This implementation is maintained on the `task1_xinyu_XGBoost` branch of
[DSA4262_Genomix](https://github.com/Yuuuu511/DSA4262_Genomix/tree/task1_xinyu_XGBoost/m6a_project).
The repository is currently private; the course requires a public repository
at final submission. Verify Ubuntu
installation before the peer evaluation. Task 2 SG-NEx analysis, the interactive
visualization, and the final report are separate deliverables. The final report
must include AI usage disclosure; a factual working record is in `AI_USAGE.md`.
