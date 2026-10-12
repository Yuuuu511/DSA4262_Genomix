# Genomix_haribo — Task 1 m6A prediction

This package uses the selected **data0-only XGBoost model** evaluated on the original RF gene split. It includes the trained model, a training script, a prediction script, a small prediction input, a fixed split manifest, and evaluation records. Prediction needs only a JSON file and the model; it needs no labels, gene mappings, raw training data or split file.

## Install

Use Python 3.11–3.13. The provided model and scripts were verified on Python 3.13/macOS, with the versions pinned in requirements.txt. The user also ran all four bundled pipeline tests successfully on an AWS Ubuntu 24.04 x86_64 instance with a Python 3.11 XGBoost environment. Official m6Anet inference and all three benchmark comparisons were run on that instance in a separate Python 3.8 environment. On an Ubuntu machine with Python and venv available, or on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

XGBoost depends on an OpenMP runtime. If Ubuntu reports a missing `libgomp.so.1`, install the system package `libgomp1`. Load only the included trusted model or a model you trained yourself.

## Predict immediately

From this folder:

```bash
python predict.py --data example/test.json.gz --output example/predictions.csv
python validate_submission.py --data example/test.json.gz --predictions example/predictions.csv
python test_pipeline.py
```

The first command defaults to `artifacts/selected_xgb.joblib`. The included input contains 20 sites; `example/expected_predictions.csv` was freshly generated with this chosen model. Tests run inference again and compare keys and scores, including different batch sizes. Probability tolerance is 1e-6 with the pinned versions.

For new data:

```bash
python predict.py --data /path/to/new_data.json.gz --model artifacts/selected_xgb.joblib --output /path/to/predictions.csv
```

Plain newline-delimited `.json` and `.json.gz` are supported. Output contains every input site, with exactly these columns:

```csv
transcript_id,transcript_position,score
```

Scores are finite probabilities in [0,1]. There is no label threshold and no CSV index. Transcript identifiers and integer positions are preserved, including position 0. Duplicate sites or malformed read arrays cause an error. Prediction processes feature batches; identifiers are retained to detect duplicates. Existing output is refused unless `--overwrite` is explicitly passed. Partial predictions are not left at the requested output path after an error.

## Generate a gene split without any RF code

`prepare_split.py` reads label metadata directly. It uses GroupShuffleSplit with seed 42 for the 20% test genes and seed 43 for validation (25% of remaining genes). The fractions refer to genes, not exact site counts. The split is independent of which classifier will be trained. It does not import RF or train_model.py.

For a standalone data0-only XGBoost experiment:

```bash
python prepare_split.py --data0-labels ../data0/data.info.labelled --output splits/data0_only_manifest.csv --check-against splits/split_manifest.csv
python train.py --data ../data0/dataset0.json.gz --labels ../data0/data.info.labelled --manifest splits/data0_only_manifest.csv --output-dir artifacts/data0_retrained
```

For the course data, the data0-only command was verified to reproduce the current data0 assignments exactly. It saves only data0 rows, so this new model's saved manifest will not contain data1 evaluation rows. The current evaluate_test.py requires the full data0+data1 manifest; use the full reproduction route below for that two-source evaluation. Prediction on new data works with either model and does not need any manifest.

To reproduce the **entire current manifest**, including data1 held-out evaluation rows:

```bash
python prepare_split.py --data0-labels ../data0/data.info.labelled --data1-labels ../data1/data.info --output splits/reproduced_manifest.csv --check-against splits/split_manifest.csv
```

This exact reproduction was tested on the real course label files, including row order and all split assignments. Data1 transcript-to-gene mappings are obtained from data0 labels when missing. The 15,137 data1 sites without a mapped gene are excluded, matching the original experiment. Data2 is not used in splitting. The script reads data1 labels for metadata integrity and class checks; no data1 observations enter XGBoost fitting, early stopping or configuration selection.

The original manifest was first generated in a shared experiment used to compare RF and XGBoost. This script independently implements that actual preparation logic without the RF model code. Preserve that provenance in the report.

Optional `--data0-json` and `--data1-json` arguments verify exact raw JSON/label site coverage. No signal features are needed to assign genes to splits. Training separately checks raw feature coverage for the requested data0 training/validation sites. Existing manifest outputs are never overwritten.

## Training from raw JSON and labels

Training uses `train.py` and the bundled original `splits/split_manifest.csv`. It accepts the course data0 JSON and binary label CSV. Labels must contain `gene_id,transcript_id,transcript_position,label`; the script verifies they match the manifest. It reads labels for identity checks but uses only data0 train/validation labels for fitting and selection.

If this folder is inside your existing Group_Project, the data files remain in sibling data0/data1/data2 directories:

```bash
python train.py --data ../data0/dataset0.json.gz --labels ../data0/data.info.labelled --output-dir artifacts/retrained
```

On another machine, replace these paths with the downloaded course training files. Raw training datasets are not bundled. A separate training example is not required by Task 1.

No training is needed to use the included model. Retraining defaults to a new `artifacts/retrained` folder and does not replace the included model. To predict with a retrained model, pass its path explicitly:

```bash
python predict.py --data /path/to/new_data.json.gz --model artifacts/retrained/selected_xgb.joblib --output /path/to/new_predictions.csv
```

The bundled manifest is specific to this experiment. Supplying unrelated data will fail coverage checks rather than silently selecting another split. To use a different dataset, prepare a new manifest with the same schema and globally disjoint genes/sites and pass `--manifest`; the source name of the training observations must be `data0`.

## Fixed evaluation protocol

- Candidate training: data0 train, 73,233 sites.
- Validation: data0 validation, 24,773 sites.
- Final model fit: these two partitions only, 98,006 sites.
- Test: original data0+data1 test, 39,019 sites (23,832 data0; 15,187 data1).

Genes and transcript-position sites cannot cross manifest splits. Training does not compute test features. Four candidates use depth 3/5 and positive-class weight 1/4, with validation early stopping. The mean of validation ROC-AUC and trapezoidal PR-AUC selects a configuration. The selected configuration is depth 5, weight 1, 408 boosting rounds. Seed 4262 and the original XGBoost settings are retained. The model is **not** refitted on all data0.

Features are the unchanged original Group_Project 110 features: signal summaries (mean, standard deviation, quantiles), log dwell summaries, center/flank contrast summaries, read count, sequence one-hot features and motif indicators. Identifiers and labels are not model features. The analytic weight correction is retained; selected weight 1 means it is the identity. Probabilities are not empirically calibrated.

To reproduce held-out evaluation without retraining:

```bash
python evaluate_test.py --data0 ../data0/dataset0.json.gz --data1 ../data1/dataset1.json.gz --output-dir evaluation/recomputed
```

This command verifies the model's saved manifest matches the evaluation manifest and never calls fit. Existing evaluation output is refused. The measured held-out results are in `evaluation/test_metrics.csv`; the final m6Anet comparison is in `evaluation/m6anet_comparison/comparison_metrics.csv`. See RESULTS.md for interpretation. These test results have already been viewed during method comparison; model choice using them is exploratory, not a new independent evaluation.

## Generate three course prediction files

```bash
python run_pipeline.py --data-root .. --output-dir submissions --team-name Genomix_haribo
```

This wrapper predicts and checks exact coverage using the included frozen model. It **does not retrain**. It expects data0/dataset0.json.gz, data1/dataset1.json.gz and data2/dataset2.json.gz under the data root. Check registered team spelling before submission. Existing prediction files are refused; select a new output directory to run again. No files are uploaded automatically.

## Benchmark against official m6Anet on Ubuntu x86_64

This complete workflow uses the sibling course data folders only. No other project directory or earlier prediction files are required. Official m6Anet 2.0.2 was run on AWS Ubuntu 24.04 x86_64 in its own Python 3.8 environment. XGBoost prediction and metric scoring use the main environment. The full course datasets are available through the course data distribution (Canvas); they are not included in this public repository.

Install a separate m6Anet environment using conda, as requested by the course:

```bash
conda create -n m6anet38 --override-channels -c pytorch -c conda-forge -c bioconda -c defaults python=3.8 m6anet=2.0.2 cpuonly "numpy<1.24"
```

Prepare inputs in the main environment:

```bash
python prepare_m6anet.py --data0 ../data0/dataset0.json.gz --data1 ../data1/dataset1.json.gz --output-dir evaluation/m6anet_inputs
python prepare_m6anet_data2.py
```

Data0/data1 preparation selects original test sites; data2 preparation uses all raw data2 sites. Both append synthetic per-site read IDs to the course's nine-column signals, rebuild byte offsets and retain sites with at least 20 reads. All 39,019 data0/data1 test sites and all 1,323 data2 sites passed that coverage threshold in this run. Preparation needs an empty/absent destination; intermediate JSON files are intentionally excluded from Git.

Run official inference separately for all three sources, using the package's default HCT116 model and normalization:

```bash
conda activate m6anet38
m6anet inference --input_dir evaluation/m6anet_inputs/data0 --out_dir evaluation/m6anet_outputs/data0 --n_processes 1 --num_iterations 1000 --seed 42 --save_per_batch 100000
m6anet inference --input_dir evaluation/m6anet_inputs/data1 --out_dir evaluation/m6anet_outputs/data1 --n_processes 1 --num_iterations 1000 --seed 42 --save_per_batch 100000
m6anet inference --input_dir evaluation/m6anet_inputs/data2 --out_dir evaluation/m6anet_outputs/data2 --n_processes 1 --num_iterations 1000 --seed 42 --save_per_batch 100000
```

**Version-specific saving workaround:** the installed 2.0.2 code uses `if (it + 1) % args.save_per_batch` and lacks a final flush. The default value 2 omitted the last 8 data0 sites and last 3 data1 sites. Setting 100000, greater than the number of batches in each prepared dataset, makes that condition true for every batch and restores complete coverage. Do not set it to 1; in this version that prevents saving. This is a workaround for 2.0.2, not a universally appropriate parameter for future versions. Model weights and feature normalization are not altered. Record the workaround with the benchmark.

Return to the main XGBoost environment and score:

```bash
conda activate xgb
python benchmark_m6anet.py --m6anet-data0 evaluation/m6anet_outputs/data0/data.site_proba.csv --m6anet-data1 evaluation/m6anet_outputs/data1/data.site_proba.csv --output-dir evaluation/m6anet_recomputed
```

The original results in evaluation/m6anet_comparison are retained for reporting; this reproduction command uses a separate output location so it cannot overwrite them. Both methods are evaluated on exactly the same test sites, and missing coverage is reported.

### Data2 label convention and separate comparison

The evaluation convention agreed for this project is `label >= 0.5 -> 1`, otherwise 0. Keep fractional labels for provenance. There are 189 values exactly 0.5; all are assigned to class 1. No data2 labels are used for fitting or tuning. Data2 has a positive rate of 71.43%, so its PR-AUC/AP should not be directly ranked against the much lower-prevalence data0/data1 results.

```bash
python evaluate_data2.py --m6anet-predictions evaluation/m6anet_outputs/data2/data.site_proba.csv --output-dir evaluation/data2_recomputed
```

By default this reads submissions/Genomix_haribo_dataset2.csv, generated by run_pipeline.py. If that file has not been generated, run the prediction pipeline first or supply --predictions with its location. The scorer compares common sites and retains original labels; it does not threshold prediction probabilities. The recorded comparison is in evaluation/data2_evaluation.

The repository includes the compact site-level m6Anet CSVs used for these comparisons. Large data.json and data.indiv_proba.csv intermediates are excluded; the commands above reproduce them. Pretrained m6Anet overlap with course data is unknown. Our model summarizes signals into fixed site features; m6Anet is a neural multiple-instance method with read-level inference and sampling-based site pooling. No significance test was performed.

Official sources: [m6Anet](https://github.com/GoekeLab/m6anet), [Bioconda package](https://bioconda.github.io/recipes/m6anet/README.html). Cite the m6Anet authors in the project report: Hendra et al. (2022), [Nature Methods](https://doi.org/10.1038/s41592-022-01666-1).

## Files

- prepare_split.py: independent gene split generation from label metadata.
- train.py / predict.py: the two core Task 1 entry points.
- xgb_features.py: shared feature implementation; keep unchanged unless retraining.
- artifacts/selected_xgb.joblib: the exact selected model, not the old all-data0 model.
- splits/split_manifest.csv: the original RF split required for training/evaluation reproduction.
- example/: input and newly generated expected predictions.
- run_pipeline.py / validate_submission.py / test_pipeline.py: batch prediction and checks.
- evaluate_test.py / prepare_m6anet.py / benchmark_m6anet.py: held-out evaluation tools.
- prepare_m6anet_data2.py / evaluate_data2.py: separate data2 preparation and thresholded-label comparison.
- evaluation/: final XGBoost results, official m6Anet site scores, complete data0/data1 comparison and separate data2 comparison.
- PROVENANCE.json / AI_USAGE.md: package and assistance records.

The folder contains no old logistic-regression training code, old model.joblib, old submissions or local editor settings. RF results are retained only as comparison records. Keep the original project folder as an archive. Publish the replacement folder in a public GitHub repository before submission, and retain the AWS Ubuntu verification record for peer evaluation. No repository has been created or published as part of this packaging.

## Remaining report work

The course also requests a simple-baseline comparison. This package currently records the final XGBoost and official m6Anet comparisons; a simple-baseline result on the same independent test split still needs to be included or reproduced before claiming that all report requirements are complete. No GitHub publication or course submission is performed by these scripts.
