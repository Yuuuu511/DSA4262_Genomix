# Final measured results — 12 October 2026

The final model is data0-only XGBoost, selected using data0 validation and refitted on data0 train+validation (98,006 sites). Original test genes never entered fitting or early stopping. The model has not been refitted on all data0. Current model selection also considered previously viewed test comparisons and must be described as exploratory.

Official m6Anet 2.0.2 inference was run by the user on AWS Ubuntu 24.04 x86_64, using Python 3.8, PyTorch 1.6.0, default HCT116 weights/normalization, 1,000 sampling iterations, seed 42 and one worker. The version-specific `--save_per_batch 100000` workaround restored the previously omitted final batches. These results supersede the incomplete 39,008-site comparison and older predictions from other experiments.

| Method | Evaluation dataset | Sites | Positive rate | ROC-AUC | PR-AUC | AP |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | data0+data1 test | 39,019 | 6.02% | 0.876746 | 0.462438 | 0.462871 |
| m6Anet | data0+data1 test | 39,019 | 6.02% | 0.878555 | 0.446030 | 0.446313 |
| XGBoost | data0 test | 23,832 | 4.97% | 0.926827 | 0.530420 | 0.531173 |
| m6Anet | data0 test | 23,832 | 4.97% | 0.935117 | 0.527973 | 0.528455 |
| XGBoost | data1 test | 15,187 | 7.67% | 0.820503 | 0.399059 | 0.399623 |
| m6Anet | data1 test | 15,187 | 7.67% | 0.813147 | 0.369707 | 0.370311 |
| XGBoost | all data2 | 1,323 | 71.43% | 0.704471 | 0.860361 | 0.860479 |
| m6Anet | all data2 | 1,323 | 71.43% | 0.673856 | 0.864178 | 0.864272 |

PR-AUC is trapezoidal integration of the precision-recall curve; AP uses average_precision_score. All comparisons in a given row pair use identical sites. All original held-out data0/data1 sites and all data2 sites are covered; no m6Anet sites were missing in the final comparison.

On data0/data1 test, XGBoost has higher PR-AUC/AP in each source and overall. m6Anet has higher overall and data0 ROC-AUC. On data2, XGBoost has higher ROC-AUC while m6Anet has slightly higher PR-AUC/AP. Neither method dominates every metric. No confidence interval or significance test was performed.

## Data2 interpretation

Original fractional labels are preserved. For evaluation, values >= 0.5 are class 1 and values < 0.5 are class 0. The 189 exact 0.5 values were assigned to class 1 by the explicit project convention agreed by the user; that boundary should not be attributed to a separately verified instructor instruction. There are 945 positive and 378 negative sites under this rule. Predictions remain continuous probabilities, and labels were not used to tune either model.

Data2's prevalence and label definition differ from data0/data1, so its high PR-AUC/AP must not be interpreted as directly better generalization than the other sources. It is reported separately rather than merged into one overall test figure.

## Verification and limitations

The user reproduced the XGBoost training and held-out results on macOS, generated and validated all three course submission CSVs, ran all four pipeline tests successfully on AWS Ubuntu, and executed official m6Anet inference/scoring there. Full XGBoost retraining on AWS was not reported; AWS verification covered saved-model prediction tests and benchmark execution.

The gene split was originally created for a shared RF/XGBoost experiment. prepare_split.py reproduces it independently from label metadata; no RF model code is required. Only data0 train/validation is used by the supplied XGBoost training script. Data1/data2 labels are not used for model tuning. Pretrained m6Anet training overlap with the course data is unknown.

A simple-baseline comparison is still required for the final report unless the team includes a suitable, separately documented baseline result. This package does not claim to complete that requirement. New independent test data is needed for an unbiased estimate after choosing methods based on the current test results.

## Result files

- evaluation/test_metrics.csv: XGBoost on the original held-out set.
- evaluation/m6anet_comparison/comparison_metrics.csv and coverage.json: final complete data0/data1 benchmark.
- evaluation/data2_evaluation/data2_metrics.csv and data2_evaluation_report.json: separate data2 evaluation.
- evaluation/m6anet_outputs/data*/data.site_proba.csv: site-level official predictions used for scoring.

Large prepared JSON and individual-read prediction files can be regenerated and are not included in Git.
