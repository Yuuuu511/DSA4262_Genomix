# Training on data0, data1 and data2

Run: 2026-10-06T10:48:37+08:00. Model: standardised L2 logistic regression with the same 45 features as the previous baselines. User-selected data2 label rule: **positive when the supplied value is greater than or equal to 0.5; negative otherwise**.

## Outcome

All **213,971 observations** from the three datasets were included. An 80/20 group split reserved the test observations before five-fold validation and parameter selection. Cross-validation selected **C = 10**. The model was then fitted on **169,525 training observations** and evaluated once on **44,446 test observations**. The saved model has not been refitted using the test set.

## Label treatment

Data0 and data1 retain their binary labels. Data2's raw labels are converted inclusively at the requested 0.5 boundary:

| Raw data2 label | Training label | Observations |
|---|---:|---:|
| 1.0 | 1 | 189 |
| 0.95 | 1 | 189 |
| 0.75 | 1 | 189 |
| 0.7 | 1 | 189 |
| 0.5 | 1 | 189 |
| 0.25 | 0 | 189 |
| 0.0 | 0 | 189 |

This yields **945 positive and 378 negative data2 observations**. Raw values are retained in `features.npz`, `split.csv`, `test_predictions.csv` and `oof_predictions.csv`; the separate `label` field is the binary training target. There is no integer truncation of fractional values. This threshold defines the data2 class used for training and evaluation; it does not make the model an estimator of the original fractional value. The biological definition of those fractions is not established by the supplied handout.

## Grouping and workflow

1. Reuse the original, label-free 45 features: nine across-read means, nine across-read standard deviations, log read count, 18 central-motif indicators and eight outer-base indicators. Dataset/condition identifiers are audit metadata, not input features.
2. For data0 and data1, retain the **4,379 conservative gene groups** resolved during [the previous combined training](combined_training.md). Gene aliases and shared transcript-position coordinates stay together. The bundled compact mapping preserves the original annotation unions; no network lookups are required.
3. For data2, group all seven condition versions of each position together. There are **189 source-position groups**. The same position has an identical sequence context across conditions, and 95.05% of unique read vectors in the 0.95 condition also occur in the 1.0 condition. Splitting these versions independently would reuse measurements across training and testing.
4. Split the resulting **4,568 groups** with `GroupShuffleSplit`, seed 42, and test size 0.2. Training contains 3,654 groups and testing contains 914. Group sizes differ, so observations are divided **79.23% / 20.77%**, rather than exactly 80/20.
5. Use five-fold `StratifiedGroupKFold`, seed 43, only within outer training. For each C candidate, fit a fresh scaler and model on four folds and predict the fifth. No held-out test labels are used for candidate selection.
6. Choose C by mean validation average precision. Refit one scaler and model on the full outer training portion, then evaluate the held-out test. Fit uses one equal-weight observation per dataset/site; no class weights or dataset balancing are added.

Data2 has no biological gene mapping. Its seven `tx_id_*` identifiers represent the repeated label conditions in the files, so they can occur in both partitions while their source positions cannot. This data2 grouping establishes separation of repeated source positions, not independence of biological genes. The legacy `gene_group` CSV field contains either a natural-data gene group or a namespaced data2 position group; data2's `gene_id` field is blank.

The preparation audit found no natural-data sites sharing at least three selected read anchors with data2. That limited comparison sampled up to 30 unique vectors per condition and position and does not prove that the complete datasets have independent biological origins. See [the preparation findings](data2_training_preparation.md).

## Split counts

| Partition | Observations | Positives | Positive fraction | Natural gene groups | Data2 source-position groups |
|---|---:|---:|---:|---:|---:|
| all | 213,971 | 13,013 | 6.08% | 4,379 | 189 |
| train | 169,525 | 10,388 | 6.13% | 3,507 | 147 |
| test | 44,446 | 2,625 | 5.91% | 872 | 42 |

| Dataset | All observations | Training | Test |
|---|---:|---:|---:|
| data0 | 121,838 | 96,282 | 25,556 |
| data1 | 90,810 | 72,214 | 18,596 |
| data2 | 1,323 | 1,029 | 294 |

Data2 contributes 1,029 training observations from 147 positions and 294 test observations from 42 positions. Each position retains all seven conditions in its partition. Across all sources there are 146,651 distinct transcript-position coordinates; the 67,320 shared data0/data1 coordinates remain separate dataset observations with their original labels. Their 3,872 conflicting label pairs are retained, not reconciled.

## Cross-validation

Means ± sample standard deviations over the five folds are shown below. Standard deviations describe fold variation, not confidence intervals.

| C | Average precision, mean ± SD | ROC AUC, mean ± SD |
|---:|---:|---:|
| 0.01 | 0.389462 ± 0.019713 | 0.857302 ± 0.007883 |
| 0.1 | 0.414015 ± 0.021687 | 0.861787 ± 0.007573 |
| 1 | 0.416413 ± 0.022181 | 0.861573 ± 0.007396 |
| 10 | 0.416612 ± 0.022233 | 0.861547 ± 0.007364 |

The selection rule chose **C = 10**. Its mean AP advantage over C = 1 is only 0.000199, which is small relative to between-fold variation. Validation scores used for selection can be optimistic; the reserved test is the separate evaluation for this run.

| Fold | Fitting observations | Validation observations | ROC AUC | Average precision |
|---|---:|---:|---:|---:|
| 1 | 135,589 | 33,936 | 0.851071 | 0.392275 |
| 2 | 135,638 | 33,887 | 0.867962 | 0.443565 |
| 3 | 135,637 | 33,888 | 0.869045 | 0.435774 |
| 4 | 135,622 | 33,903 | 0.861222 | 0.410927 |
| 5 | 135,614 | 33,911 | 0.858433 | 0.400519 |

## Combined held-out results

| Metric | Selected-C cross-validation mean ± SD | Held-out test |
|---|---:|---:|
| ROC AUC | 0.861547 ± 0.007364 | 0.858137 |
| Average precision | 0.416612 ± 0.022233 | 0.414077 |
| Trapezoidal PR AUC | 0.416161 ± 0.022249 | 0.413727 |
| Log loss | 0.165399 ± 0.003676 | 0.162083 |
| Brier score | 0.044183 ± 0.000953 | 0.042934 |

The constant predictor assigning the training positive fraction to every test observation has ROC AUC **0.500000**, AP **0.059060**, log loss **0.224418**, and Brier score **0.055577**. AP and trapezoidal PR AUC use different integration conventions; both are reported. ROC AUC is a ranking metric, not classification accuracy.

## Results by dataset

| Test dataset | Observations | ROC AUC | Average precision | Trapezoidal PR AUC | Log loss | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| data0 | 25,556 | 0.894070 | 0.428090 | 0.427245 | 0.118535 | 0.030191 |
| data1 | 18,596 | 0.812806 | 0.340509 | 0.340021 | 0.209207 | 0.056248 |
| data2 | 294 | 0.734354 | 0.895054 | 0.894786 | 0.966798 | 0.308520 |

Data2's positive prevalence is **71.43%**, so its constant-score AP reference is **0.714286**, much higher than the natural-data reference. Its AP of 0.895054 must be read alongside ROC AUC **0.734354**. Data2's mean predicted probability is **0.366182**, compared with its observed positive fraction of 0.714286. Together with log loss **0.966798** and Brier score **0.308520**, this indicates substantial probability underestimation for this threshold-defined data2 endpoint. Natural-data observations dominate the equal-weight pooled objective.

Adding 189 groups changes natural-data test membership under the same seed. These scores are therefore not a controlled comparison against the previous data0/data1 model, and they do not demonstrate that adding data2 caused an improvement. Previous project runs already examined the source datasets; this holdout is independent of fitting and selection for this run, not a completely untouched project-level dataset. Performance on the hidden leaderboard remains unknown. The mixed label definitions and uncertain data2 biological provenance limit interpretation of the pooled score.

## Verification

**Nine automated tests passed.** A separate audit confirmed the threshold boundary, preservation of raw labels, grouping of repeated data2 positions, complete CV prediction coverage, no outer or fold group overlap, and training-only scaler statistics. Exported test and fold metrics reproduce the recorded results. The ordinary inference script reproduced predictions for **60 real held-out observations** across all three sources, with maximum absolute error **4.44e-16**. No test observations were included in the saved model's fit.

Published numeric results and historical checks are in `results/all_datasets/metrics.json` and `results/all_datasets/verification.json`. Source file hashes and library versions are recorded there.

## Reproduce training and make predictions

From the project directory with the existing environment activated:

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

With data2 supplied, the default output becomes `artifacts/all_datasets/`. The data0/data1 feature cache is reused only when its source hashes and feature specification match; data2 signal coordinates, contexts and read counts are checked during extraction. The all-datasets feature cache also records the label threshold and grouping rule. The bundled `reference/transcript_gene_map.csv` preserves the exact gene groups. Its checksum and the two natural-data label-file hashes are verified before training; the large annotation files are not needed for the default command.

For a released leaderboard input, use the packaged model (also the default for `predict.py`):

```bash
python predict.py --data PATH_TO_LEADERBOARD_INPUT.json.gz \
  --model models/baseline_model.joblib \
  --output artifacts/all_datasets/intermediate_predictions.csv
```

The 0.5 rule applies to **training labels**, not inference scores. The submission file still contains probabilities in `transcript_id,transcript_position,score`, without thresholding. Test and OOF diagnostic files contain additional columns and should not be submitted as leaderboard prediction files.

| Output | Purpose |
|---|---|
| `artifacts/all_datasets/baseline_model.joblib` | Model fitted on the outer training portion of all three sources |
| `artifacts/all_datasets/metrics.json` | CV candidates/folds, test results by source, versions and provenance |
| `artifacts/all_datasets/split.csv` | Every observation's group, binary/raw label, outer split and validation fold |
| `artifacts/all_datasets/test_predictions.csv` | Reserved test probabilities and labels |
| `artifacts/all_datasets/oof_predictions.csv` | Validation predictions for the selected C |
| `artifacts/all_datasets/verification.json` | Artifact audit results |

The table describes regenerated local outputs. The GitHub package includes `models/baseline_model.joblib` and the summaries in `results/all_datasets/`; split assignments, feature caches and large prediction CSVs are regenerated locally. Earlier models are in `models/`, with their reports in `docs/` and historical numeric summaries in `results/`. Publication preparation checks are recorded separately in `results/repository_verification.json`.
