# Measured results

All training and model selection used data0 only. Seed: 4262.

| Method / evaluation | Sites | ROC AUC | PR AUC (trapezoid) | Average precision |
|---|---:|---:|---:|---:|
| Mean-feature logistic baseline, data0 heldout genes | 23,489 | 0.7161 | 0.1093 | 0.1097 |
| Our XGBoost, data0 heldout genes | 23,489 | 0.9280 | 0.5074 | 0.5080 |
| Pretrained m6Anet adapter, same data0 sites | 23,489 | 0.9295 | 0.5192 | 0.5198 |
| Our final model, all dataset1 sites | 90,810 | 0.8308 | 0.3905 | 0.3906 |
| Our evaluation model, dataset1 heldout genes | 14,928 | 0.8354 | 0.4013 | 0.4018 |

The test partition contains 1,127 positive sites out of 23,489 (4.798%). The random ROC AUC reference is 0.5; the no-skill precision reference is the positive prevalence. These reference values are not results from a simulated random classifier.

Selected model: depth 3, positive class weight 1, 680 boosting rounds, learning rate 0.05. Validation selection score was 0.7214. No dataset1/2 results were used for selection.

The pretrained m6Anet adapter scores slightly higher on ROC/PR AUC. Its original training overlap is unknown, so this comparison is not a claim about independent generalization. The adapter uses official weights and normalization with exact expected noisy-OR pooling, not the unmodified CLI. The simulation check differed by 0.00004856. See README for environment and pooling differences.

Dataset1 all-site results are affected by overlap with data0 genes. The heldout-gene subset has 14,928 sites and uses the evaluation model trained without those genes. Dataset1 and data0 share 67,320 site keys; 3,872 labels differ. This is a change of sample, not an independent partition of the final model.

Dataset2 uses fractional labels. No binary threshold was assumed, so no ROC/PR AUC is reported. Its descriptive Spearman association with model score is 0.2912. This does not establish probability calibration or a modification-fraction estimator.

All three submission CSVs passed exact key coverage, schema, duplicate and probability checks. See `artifacts/submission_validation.json`.

![Evaluation curves](artifacts/evaluation_curves.png)
