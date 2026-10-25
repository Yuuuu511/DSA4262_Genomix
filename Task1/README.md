# Task 1: m6A prediction with XGBoost

The default predictor uses the selected **data0+data1 XGBoost**, trained on gene-disjoint train/validation partitions. It accepts new m6Anet-processed JSON or gzip JSON without labels and emits exactly `transcript_id,transcript_position,score`. Scores remain continuous in [0,1].

The two entry points required by Task 1 are **train.py** and **predict.py**. Both use the shared, unchanged feature implementation in xgb_features.py. The remaining scripts reproduce splitting, validation, benchmarking and submission generation.

## 1. Install on AWS Ubuntu and run the example

### Requirements

Target platform: AWS Ubuntu 24.04, Linux x86_64, CPU only. Use Python 3.11 and the pinned packages in [requirements.txt](requirements.txt). A GPU is not required. Git and an internet connection are needed to download the repository and dependencies; prediction then runs locally.

The model was trained with XGBoost 2.1.4, scikit-learn 1.6.1 and NumPy 2.2.3. This reorganized package has been smoke-tested on macOS with the pinned dependencies. Its predecessor ran on AWS Ubuntu; verify this exact checkout on Ubuntu before final submission.

### Install

If conda is already installed, skip the following installer. On a fresh **Linux x86_64** server without conda, install [Miniforge](https://github.com/conda-forge/miniforge) under your home directory:

```bash
curl -fsSL -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Do not rerun the installer over an existing installation. Create a dedicated environment:

```bash
conda create -n m6a_task1 python=3.11 -y
conda activate m6a_task1

git clone --branch main https://github.com/Yuuuu511/DSA4262_Genomix.git
cd DSA4262_Genomix/Task1
python -m pip install -r requirements.txt
```

If the repository is already downloaded, enter its `Task1` directory instead of cloning it again. Private repositories require GitHub authentication; the course submission repository must be publicly accessible by the deadline.

### Quick prediction check — no training required

Run these commands from `Task1/`:

```bash
python test_pipeline.py
python predict.py --data example/test.json.gz --output example/predictions.csv
python validate_submission.py --data example/test.json.gz --predictions example/predictions.csv
```

Expected: **4 tests OK**, **20 predictions**, and validation **PASS**. The bundled compressed example is approximately 60 KiB, below 1 MB, and contains no labels. [expected_predictions.csv](example/expected_predictions.csv) contains reference model scores, not ground truth. You do not need the training datasets or m6Anet installed to run this example.

Existing prediction files are protected. For an intentional repeat, add `--overwrite` to `predict.py`, then rerun the validator. Use `python predict.py --help` for usage.

## 2. Predict unseen data

```bash
python predict.py --data /path/to/new_dataset.json.gz --output predictions.csv
```

The default model is `artifacts/selected_xgb.joblib`, the NEW joint model. It requires no labels, gene IDs, training split or model retraining. Plain `.json` and `.json.gz` are supported. Every input site is scored, including sites below 20 reads; malformed/duplicate sites fail explicitly instead of being silently dropped. Position zero is supported. IDs and transcript positions are preserved; positions are not converted to genomic coordinates.

```bash
# Explicitly select a separately retrained model:
python predict.py --data /path/to/new.json.gz \
  --model artifacts/retrained/selected_xgb.joblib --output new_scores.csv
```

Input uses m6Anet-processed, newline-delimited JSON: each site is nested as `transcript_id → transcript_position → sequence → reads`. Each read contains nine numeric signal features. Supply this processed file, not FASTQ or raw nanopore signal files.

Output is a CSV with a header and one row per input site:

```text
transcript_id,transcript_position,score
ENST00000007516,186,0.0784527063369751
```

`score` is the model's continuous m6A prediction score; larger values indicate stronger model support. The identifier and position refer to the input transcript, not a genomic chromosome coordinate.

Only scores are output. A 0.5 binary decision is not applied to submission scores. These model probabilities have not been empirically calibrated as modification fractions.

## 3. Local training-data layout

Raw data and labels are local inputs, not part of the public code package:

```text
Group_Project/
  data0/dataset0.json.gz
  data0/data.info.labelled
  data1/dataset1.json.gz
  data1/data.info
  data2/dataset2.json.gz
  data2/data.info
  Task1/
    train.py
    predict.py
    xgb_features.py
    artifacts/selected_xgb.joblib
    splits/gene_split.csv
```

Default paths follow this layout relative to the scripts; they do not depend on a particular Mac username, AWS home path, or terminal working directory. If you unpack Task1 into Downloads, prediction/tests still work, and training/evaluation can use explicit input paths.

## 4. Train a new joint model

```bash
python train.py --check-only
python -u train.py
```

Default training output is `artifacts/retrained/`, protecting the packaged selected model. To choose input paths explicitly:

```bash
python -u train.py \
  --data0 ../data0/dataset0.json.gz --labels0 ../data0/data.info.labelled \
  --data1 ../data1/dataset1.json.gz --labels1 ../data1/data.info \
  --manifest splits/gene_split.csv \
  --output-dir artifacts/retrained
```

Four candidates: max_depth in {3,5}, scale_pos_weight in {1,4}. Other settings: learning_rate=0.05, max rounds=1200, early stopping=60, min_child_weight=15, subsample=0.85, colsample_bytree=0.85, reg_lambda=10, reg_alpha=0.1, hist trees, seed=4262. Default n_jobs=2; override with --n-jobs if appropriate.

Candidates fit only both datasets' TRAIN observations. Early stopping uses combined VALIDATION aucpr. Configuration selection maximizes `(combined validation ROC-AUC + combined validation trapezoidal PR-AUC)/2`. The selected round count/configuration is refitted on both datasets' TRAIN+VALIDATION observations. Test is not used for feature aggregation, fitting, early stopping or selection. Streaming the JSON files to find requested keys does not compute test-site features.

The bundled selected model has depth=3, positive weight=1 and 912 boosting rounds. The fitted model was trained on macOS. The packaged bundle omits label-bearing metadata; its fitted booster and feature implementation are unchanged.

Training writes the model, parameters, validation metrics overall/per dataset and local audit tables. Label-bearing audit/prediction tables generated locally are gitignored and are not part of this package. Nonempty output directories are refused; choose a new --output-dir for another run. No automatic resume is implemented for partial model fits.

## 5. Gene split and exclusions

The public split file contains **gene_id,split only**, not ground-truth site labels. The code reconstructs site membership from your local label tables and verifies that genes/transcripts/site coordinates do not cross splits. Observations are keyed by source_dataset + transcript_id + transcript_position; cross-source observations with identical coordinates retain their own labels.

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| data0 | 73,233 | 24,773 | 23,832 |
| data1 | 45,146 | 15,340 | 15,187 |
| Combined | 118,379 | 40,113 | 39,019 |

The final model fits 158,492 train+validation observations. data1 has no gene_id column in the supplied table; shared transcript-to-gene mapping comes from data0. Exactly **15,137 data1 sites lack that mapping and remain excluded**. They are not assigned to an arbitrary unknown group. Inference still predicts all data1 input sites.

The assignments use GroupShuffleSplit: 20% of genes held out with seed 42; validation is 25% of remaining genes with seed 43. Site proportions need not be exactly 60/20/20. Gene assignments were reproduced independently of any RF model. To verify reproduction without replacing the bundled file:

```bash
python prepare_split.py \
  --data0-labels ../data0/data.info.labelled --data1-labels ../data1/data.info \
  --output splits/reproduced_gene_split.csv --check-against splits/gene_split.csv
```

For genuinely different training data, generate an appropriate new gene split explicitly and pass it with --manifest. Do not pick split seeds based on test scores. Training assumes the supplied label tables match their JSONs for retained manifest sites; missing requested features and duplicate requested raw keys fail.

## 6. Held-out test reporting

```bash
# Evaluate the bundled selected model, no fitting:
python evaluate_test.py --data0 ../data0/dataset0.json.gz --data1 ../data1/dataset1.json.gz

# For a model YOU retrained, explicitly select it:
python evaluate_test.py --model artifacts/retrained/selected_xgb.joblib \
  --evaluation-dir evaluation/recomputed_retrained
```

Input labels are needed for evaluation, but not for prediction. Outputs default to evaluation/recomputed/, protecting the reference metrics in evaluation/test_metrics.csv. Report ROC-AUC, trapezoidal PR-AUC and AP separately overall/data0/data1. See [RESULTS.md](RESULTS.md).

The original test partition has already been inspected during development. These results are exploratory reporting, not a fresh untouched final test. Do not keep tuning on them. The four additional unlabelled final datasets are for frozen-model prediction, subject to course rules.

## 7. Simple baseline

An auxiliary fixed logistic baseline uses just nine signal means, with StandardScaler fitted inside a pipeline, C=1 and class_weight=balanced. It uses exactly the same joint-source split; it is not a second candidate submission model.

```bash
python train_baseline.py
python train_baseline.py --evaluate-test
```

Default output: artifacts/retrained_baseline/. The reference baseline and its measured metrics are bundled separately as artifacts/baseline.joblib and evaluation/baseline_test_metrics.csv. It is separate from the default XGBoost predictor.

## 8. Benchmark against official pretrained m6Anet

Use a separate **Python 3.8 conda environment** on Linux x86_64. Do not install m6Anet with pip and do not mix this environment with the XGBoost prediction environment.

```bash
conda create -n m6anet38 --override-channels \
  -c pytorch -c conda-forge -c bioconda -c defaults \
  python=3.8 m6anet=2.0.2 cpuonly 'numpy<1.24' -y
```

In the XGBoost environment, prepare original test observations:

```bash
python prepare_m6anet.py \
  --data0 ../data0/dataset0.json.gz --data1 ../data1/dataset1.json.gz \
  --output-dir evaluation/m6anet_inputs
```

The adapter writes byte offsets into the newly generated data.json, appends synthetic local read IDs and applies m6Anet's minimum 20-read filter. It does not change signal features or train m6Anet.

```bash
conda activate m6anet38
for source in data0 data1
do
  m6anet inference \
    --input_dir "evaluation/m6anet_inputs/$source" \
    --out_dir "evaluation/m6anet_outputs/$source" \
    --n_processes 1 --num_iterations 1000 --seed 42 \
    --save_per_batch 100000
done
```

The unusually large save_per_batch value is a version-specific workaround for the observed m6Anet 2.0.2 final-batch saving issue. The value must exceed the number of inference batches for this dataset; it does not change weights. Coverage is checked rather than assuming inference succeeded for every site. Do not substitute individual-read probabilities or mod_ratio for site-level probability_modified.

Return to the XGBoost environment, create held-out XGBoost predictions with evaluate_test.py if necessary, and align by exact source/transcript/position:

```bash
conda activate m6a_task1  # Or the existing AWS xgb environment.
python benchmark_m6anet.py \
  --m6anet-data0 evaluation/m6anet_outputs/data0/data.site_proba.csv \
  --m6anet-data1 evaluation/m6anet_outputs/data1/data.site_proba.csv
```

The scorer defaults to evaluation/recomputed/heldout_test_predictions.csv and writes evaluation/recomputed_m6anet/. Missing scores are excluded from BOTH methods' common-site comparison, never filled with zero; missing/extra counts are reported. The included reference comparison covers 39,019/39,019 sites, using official inference files previously generated on AWS. No new m6Anet inference was run during this package preparation.

Interpretation: XGBoost aggregates per-read signal distributions into 110 site features and learns supervised site labels from data0/data1. m6Anet uses a pretrained read-based multiple-instance model and its own normalization/aggregation. Feature choices, training data, coverage rules and training provenance differ. Pretrained m6Anet training overlap with these observations is unknown, so this is an available-tool benchmark rather than proof of independent matched-training superiority.

## 9. Data2 fractional labels: explicit rule only

Data2 is not used to train or tune the model. Its fractional labels need a course-approved binary convention. There is NO silent default:

```bash
# Previously used exploratory convention:
python evaluate_data2.py --label-rule ge0.5

# Alternative convention that reproduces the historical leaderboard's Kopi
# scores/prevalence; not officially confirmed by this package:
python evaluate_data2.py --label-rule eq1 --output-dir evaluation/recomputed_data2_eq1
```

These commands expect submissions/Genomix_haribo_dataset2.csv, generated below. Optional --m6anet-predictions accepts the official data2 data.site_proba.csv. prepare_m6anet_data2.py prepares all data2 sites separately. Confirm the official rule; do not choose a convention to favour a model. Submission scores remain continuous regardless of evaluation convention.

## 10. Generate leaderboard files

```bash
python run_pipeline.py --data-root .. --output-dir submissions --team-name Genomix_haribo
```

Expected full-source row counts: 121,838 data0; 90,810 data1; 1,323 data2. CSV names follow the supplied team-name argument; Kopi is a leaderboard code name, not automatically the team's submission filename. Confirm the registered team name and current course filename instructions.

For the four new unlabelled datasets, run predict.py once per input with an appropriate unique output filename. They are not used in training, preprocessing fits, split selection or hyperparameter selection. Neither original submission scores on fitted sites nor high scores on new inputs establish true modification rates.

## 11. License

**License selection is pending team agreement.** This checkout does not currently include a `LICENSE` file and does not grant an open-source license. Before public submission, agree on a license, add its complete text at the repository root, and replace this paragraph with its name and a link to that file. Do not apply the team's code license to third-party software or datasets; their own terms apply.

## 12. Citation and software references

When referring to this project, cite the repository and the exact commit used:

> DSA4262 Genomix team (2026). *DSA4262_Genomix: m6A prediction from direct RNA sequencing*. https://github.com/Yuuuu511/DSA4262_Genomix. Commit: replace with the commit actually used.

No DOI or formal release identifier has been assigned here. Include the model training scope, dependency versions, split protocol and evaluation population when reporting results. Cite third-party methods separately using their official citation guidance: [XGBoost](https://github.com/dmlc/xgboost), [scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn), and [m6Anet](https://github.com/GoekeLab/m6anet). Document AI assistance according to the course rules.
