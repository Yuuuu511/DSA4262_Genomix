# Task 1: m6A Site Prediction Using XGBoost

This project uses an XGBoost classifier to predict whether an RNA site contains an m6A modification.

Only **data0** is used for training and model selection. The final model generates predictions for dataset0, dataset1 and dataset2. Dataset1 and dataset2 labels are not used for training or tuning.

## Model and Features

The model predicts one score for each transcript-position site by combining signals from its reads.

The 110 input features include:

- Mean, standard deviation, 10th percentile, median and 90th percentile of the nine signal measurements.
- Statistics of log-transformed dwell times.
- Differences between central and flanking signals.
- Log-transformed read count.
- Seven-base sequence and central DRACH motif encodings.

Transcript IDs, gene IDs, positions and labels are not used as model inputs.

The selected XGBoost model uses:

| Parameter | Value |
|---|---:|
| Maximum tree depth | 3 |
| Learning rate | 0.05 |
| Boosting rounds | 680 |
| Positive-class weight | 1 |

## Training and Evaluation

Data0 is divided by **gene ID** into training, validation and test sets. No gene appears in more than one split.

| Split | Sites | Genes |
|---|---:|---:|
| Training | 73,544 | 2,306 |
| Validation | 24,805 | 771 |
| Test | 23,489 | 775 |

Four XGBoost configurations are compared using the validation set. Model selection uses the mean of validation ROC AUC and trapezoidal PR AUC.

After selecting the configuration, the evaluation model is trained on the combined training and validation sets and evaluated on the held-out test set.

A separate final submission model is then trained on **all data0** using the selected configuration.

## Verified Results

The following results use the same **23,489 held-out data0 sites**, including **1,127 positive sites**.

| Model | ROC AUC | PR AUC (trapezoidal) | Average Precision |
|---|---:|---:|---:|
| XGBoost | 0.927987 | 0.507389 | 0.508020 |
| Pretrained m6Anet adapter | 0.929523 | 0.519233 | 0.519807 |

The metrics were independently recalculated from the saved predictions. Checks confirmed that:

- Test labels match the original data0 labels.
- Test sites match the saved test split.
- Training, validation and test gene sets are disjoint.
- Both models are evaluated on the same sites.
- Reloading the saved XGBoost evaluation model reproduces its predictions.

**PR AUC and average precision are different summaries of the precision–recall curve.** Both are reported explicitly. The metric used for leaderboard reporting should follow the course evaluator’s definition.

These are local held-out results, not official leaderboard scores.

### m6Anet Comparison

The comparison uses official m6Anet 2.0.1 HCT116 pretrained weights through a custom adapter.

The adapter reads the course’s nine signal columns and computes the exact expectation of the model’s 20-read noisy-OR pooling procedure, instead of using a finite Monte Carlo approximation.

This is an adapter benchmark, rather than an unmodified official m6Anet CLI run. Overlap between m6Anet’s original training data and the course data is unknown.

## Installation

This project was tested with Python 3.13 on macOS ARM64. Ubuntu compatibility has not yet been verified.

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment using:

```powershell
.venv\Scripts\Activate.ps1
```

## Generate Predictions

If the trained model and example files are available:

```bash
python predict.py \
  --data example/test.json.gz \
  --model artifacts/model.joblib \
  --output example/predictions.csv
```

For another dataset:

```bash
python predict.py \
  --data /path/to/dataset.json.gz \
  --model artifacts/model.joblib \
  --output predictions.csv
```

Prediction accepts newline-delimited `.json` and `.json.gz` files. Labels are not required.

## Retrain and Generate All Submissions

Set `--data-root` to the folder containing `data0`, `data1` and `data2`:

```bash
python run_pipeline.py \
  --data-root /path/to/Group_Project \
  --team-name Genomix_haribo
```

To reuse an existing trained model:

```bash
python run_pipeline.py \
  --data-root /path/to/Group_Project \
  --team-name Genomix_haribo \
  --skip-training
```

For training alone:

```bash
python train.py \
  --data /path/to/Group_Project/data0/dataset0.json.gz \
  --labels /path/to/Group_Project/data0/data.info.labelled \
  --output-dir artifacts \
  --cache-dir .cache
```

## Intermediate Leaderboard Submission

The pipeline generates three CSV files in `submissions/`:

| File | Number of Sites |
|---|---:|
| Genomix_haribo_dataset0.csv | 121,838 |
| Genomix_haribo_dataset1.csv | 90,810 |
| Genomix_haribo_dataset2.csv | 1,323 |

Each CSV contains exactly these columns:

```text
transcript_id,transcript_position,score
```

Scores are finite values between 0 and 1, with exactly one prediction per input site.

Validate the submission files using:

```bash
python validate_submission.py \
  --data-root /path/to/Group_Project \
  --submission-dir submissions \
  --team-name Genomix_haribo \
  --report artifacts/submission_validation.json
```

## Main Files

| File | Purpose |
|---|---|
| features.py | Extract site-level features |
| train.py | Train, select and evaluate XGBoost |
| predict.py | Generate prediction CSVs |
| run_pipeline.py | Train and generate all three submissions |
| validate_submission.py | Check submission coverage and format |
| benchmark_m6anet.py | Generate pretrained m6Anet adapter predictions |
| score_benchmark.py | Calculate benchmark metrics |
| evaluate_external.py | Evaluate predictions on external datasets |
| requirements.txt | Python dependencies |

## Interpretation and Limitations

- The reported XGBoost test results come from the evaluation model trained without test genes.
- The final submission model uses all data0. Its dataset0 predictions are therefore in-sample.
- Dataset0 and dataset1 share some transcript-position sites, so performance on all dataset1 sites is not a fully independent generalization estimate.
- Dataset2 contains fractional labels; binary ROC AUC and PR AUC are not reported for it.
- Aggregated features predict site-level modification scores, rather than identifying modified individual reads.

