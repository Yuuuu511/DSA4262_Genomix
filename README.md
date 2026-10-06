# DSA4262_Genomix

DSA4262 Team Project — Prediction of m6A RNA modifications from direct RNA-Seq data.

A reproducible logistic regression baseline for the DSA4262 intermediate leaderboard. The current model uses **data0, data1 and data2**, an 80/20 group split and five-fold cross-validation inside training. Data2 labels **at least 0.5** are positive; lower values are negative. Inference outputs site probabilities.

The trained models, compact gene mapping, small inference example and numeric evaluation summaries are included. Full training datasets, local environments, feature caches and large diagnostic CSVs are excluded from Git.

## Install

Use **Python 3.12**. From the repository directory on macOS or Ubuntu:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pinned dependency versions match the published models.

## Predict without training

```bash
python predict.py --data example/test_data.json.gz \
  --output outputs/example_predictions.csv
```

The default is `models/baseline_model.joblib`, trained on the outer training portion of all three datasets. The example contains 20 real data0 sites and checks inference; it is not an independent evaluation set. Expected results are in `example/expected_predictions.csv`.

For the released leaderboard input:

```bash
python predict.py --data PATH_TO_LEADERBOARD_INPUT.json.gz \
  --output outputs/intermediate_predictions.csv
```

Replace the placeholder with the actual input path. Plain JSON-lines files are also supported. No labels are needed. The CSV has exactly `transcript_id,transcript_position,score`, with probabilities in [0, 1]. The data2 cutoff does not threshold prediction scores. Diagnostic training CSVs have extra columns and must not be submitted. Hidden leaderboard performance has not been measured.

## Train and test

Obtain the original course datasets separately and use this layout, or pass their actual paths:

```text
data/
  data0/dataset0.json.gz
  data0/data.info.labelled
  data1/dataset1.json.gz
  data1/data.info
  data2/dataset2.json.gz
  data2/data.info
```

```bash
python train_combined.py \
  --data0 data/data0/dataset0.json.gz \
  --labels0 data/data0/data.info.labelled \
  --data1 data/data1/dataset1.json.gz \
  --labels1 data/data1/data.info \
  --data2 data/data2/dataset2.json.gz \
  --labels2 data/data2/data.info \
  --data2-threshold 0.5
```

This writes `artifacts/all_datasets/`: model, source fingerprints, fold/test metrics, exact split assignments and validation/test predictions. Generated artifacts are ignored by Git. To predict using a new run, pass `--model artifacts/all_datasets/baseline_model.joblib`; the packaged model is not overwritten automatically.

The compact mapping in `reference/` preserves the original conservative gene groups. Training verifies its checksum and both natural-data label-file hashes. No annotation download is required. See [reference provenance](reference/README.md) to rebuild the mapping from raw annotation sources.

Omit `--data2`, `--labels2` and `--data2-threshold` to reproduce the data0/data1 experiment, which writes `artifacts/combined/`. The earlier reports contain data0 training and frozen data1 evaluation commands.

## What each step does

1. **Validate signals and labels.** Stream JSON records, match transcript-position coordinates to metadata, and reject duplicates, missing labels, invalid measurements and inconsistent read counts. Preserve fractional data2 labels alongside binary targets.
2. **Extract 45 features per site.** Use nine across-read means and nine standard deviations (18), log read count (one), central DRACH motif indicators (18) and outer-base indicators (eight). Labels, dataset identity, transcript IDs, gene IDs and positions are not model inputs.
3. **Reserve test groups.** Keep related data0/data1 genes and shared coordinates together. Keep all seven data2 conditions for each source position together. Split 80%/20% of groups, seed 42, before fitting. Group sizes make row percentages differ slightly.
4. **Tune inside training.** Use five stratified group folds, seed 43. For each C in 0.01, 0.1, 1 and 10, fit a fresh scaler and logistic regression on four folds and validate on the fifth. Choose C by mean validation average precision; the test set does not enter selection.
5. **Fit the selected model.** Fit one scaler and classifier on all outer training rows with the selected C. Save them together so inference uses the learned transformation. The latest model uses the training portion only.
6. **Test reserved groups.** Report ROC AUC, average precision, trapezoidal PR AUC, log loss and Brier score, including per-dataset results and a constant reference. Verify group separation and training-only scaler statistics.

Logistic regression learns an additive weighted score after scaling, then maps it to a probability using the sigmoid function. Smaller C gives stronger L2 regularisation. No class weighting or oversampling is used.

## Recorded results

| Experiment | Evaluation | ROC AUC | Average precision | Trapezoidal PR AUC |
|---|---|---:|---:|---:|
| Data0 only | Original held-out data0 genes | 0.902878 | 0.499075 | 0.498388 |
| Frozen full-data0 model | All data1 sites | 0.809337 | 0.334759 | 0.334600 |
| Data0 + data1 | Held-out combined gene groups | 0.848127 | 0.393503 | 0.393139 |
| **Data0 + data1 + data2** | **Held-out gene/position groups** | **0.858137** | **0.414077** | **0.413727** |

The latest experiment used 169,525 training rows and 44,446 test rows. Validation selected C = 10 with mean AP 0.416612. These experiments have different test memberships and label mixtures; this is not a controlled comparison. The frozen data1 evaluation has substantial coordinate overlap with data0.

Average precision and trapezoidal PR AUC use different integration conventions. ROC AUC measures ranking, not accuracy. Data2's threshold-defined target differs from the supplied natural-data labels. Its test prevalence is 71.43%, and predictions substantially underestimate that fraction. Source-position separation does not establish biological gene independence for data2.

- [All three datasets: labels, folds, results and limitations](docs/all_datasets_training.md)
- [Data0 + data1: annotation, overlap and grouped validation](docs/combined_training.md)
- [Frozen data0 model evaluated on data1](docs/data1_evaluation.md)
- [Original data0 workflow](docs/data0_training.md)
- [Data2 preparation audit](docs/data2_training_preparation.md)

## Files and verification

| Location | Purpose |
|---|---|
| `features.py` | Shared label-free feature extraction |
| `train.py`, `train_combined.py` | Original and grouped cross-validation training |
| `predict.py`, `evaluate.py` | Submission inference and frozen data1 evaluation |
| `models/` | Four trained pipelines, descriptions and checksums |
| `reference/` | Compact gene groups, lookup snapshot and provenance |
| `results/` | Numeric summaries and verification records |
| `example/` | Small input and expected current-model predictions |
| `tests/` | Features, validation, leakage checks and packaged inference |
| `docs/` | Experiment reports and GitHub instructions |

```bash
python -m unittest discover -s tests -v
```

GitHub Actions is configured to run tests on Ubuntu with Python 3.12 after a push. Local publication checks are in `results/repository_verification.json`; historical verification files retain their original run counts.

See [GitHub setup](docs/github_setup.md) for committing and pushing to your existing branch, and [AI assistance](docs/ai_assistance.md) for preparation provenance. No software licence has been selected. Course data and reference sources retain their own terms.
