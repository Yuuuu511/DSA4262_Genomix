# Task 1 — m6A prediction with XGBoost

This project predicts m6A modification probabilities from m6Anet-processed direct RNA-sequencing JSON files.

The two main scripts are:

- **`train.py`**: train and save an XGBoost model.
- **`predict.py`**: apply a saved model to new JSON data.

The supplied model uses **data0 train and validation observations only**. Data1 and data2 are not used for fitting or hyperparameter tuning.

Prediction requires no labels or gene mappings. You can use the supplied model without retraining.

## 1. AWS Ubuntu setup

The verified deployment environment is **Ubuntu 24.04, x86_64**, with 2 vCPUs and 8 GiB RAM. Saved-model tests and official m6Anet benchmarking passed there. Full XGBoost training was reproduced on macOS; full retraining on AWS has not yet been verified.

Use separate environments:

| Environment | Python | Purpose |
|---|---|---|
| `xgb` | 3.11 | Training, prediction and evaluation |
| `m6anet38` | 3.8 | Official m6Anet inference |

The commands below run in the **AWS Ubuntu terminal**.

Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y curl bzip2 libgomp1 git
```

If conda is not already installed:

```bash
curl -fL -o /tmp/Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Create the XGBoost environment:

```bash
conda create -n xgb -c conda-forge python=3.11 -y
conda activate xgb
```

From the repository root, enter the project and install dependencies:

```bash
cd Task1
python -m pip install -r requirements.txt
```

After reconnecting through SSH, if conda is unavailable:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate xgb
```

All remaining commands assume the current directory is `Task1/`.

Directory names are case-sensitive on Ubuntu. Use **`Task1`** consistently. Renaming the GitHub directory does not automatically rename an existing copy on AWS.

## 2. Test the supplied model

```bash
python test_pipeline.py
```

The four tests check saved-model predictions, feature/manifest compatibility, batch-size consistency, site coverage and invalid-input handling.

Run the example prediction:

```bash
python predict.py \
  --data example/test.json.gz \
  --model artifacts/selected_xgb.joblib \
  --output /tmp/m6a_example_predictions.csv

python validate_submission.py \
  --data example/test.json.gz \
  --predictions /tmp/m6a_example_predictions.csv
```

The example contains 20 sites taken from data0. It is a small execution test, not an independent performance test set.

Reference predictions are in `example/expected_predictions.csv`. The tests compare scores within an absolute tolerance of `1e-6`.

Prediction refuses existing output files. For an intentional rerun, append `--overwrite`; keep the reference predictions unchanged.

## 3. Predict new data

```bash
python predict.py \
  --data /path/to/new_data.json.gz \
  --model artifacts/selected_xgb.joblib \
  --output /path/to/predictions.csv
```

Both newline-delimited `.json` and compressed `.json.gz` are supported. Each site contains a seven-base sequence and a variable number of reads, with nine finite signal features per read.

Output format:

```csv
transcript_id,transcript_position,score
```

The output contains one probability in `[0,1]` per input site. Transcript IDs and integer positions, including position zero, are preserved. No probability threshold is applied.

The default model is `artifacts/selected_xgb.joblib`, so `--model` is optional.

## 4. Full dataset layout

Download the course datasets and arrange them beside this directory:

```text
repository/
├── data0/
│   ├── dataset0.json.gz
│   └── data.info.labelled
├── data1/
│   ├── dataset1.json.gz
│   └── data.info
├── data2/
│   ├── dataset2.json.gz
│   └── data.info
└── Task1/
```

Full datasets are not included in GitHub. They are obtained through the course data distribution. The prediction example can run without them.

## 5. Reproduce the gene split

```bash
python prepare_split.py \
  --data0-labels ../data0/data.info.labelled \
  --data1-labels ../data1/data.info \
  --output /tmp/m6a_reproduced_manifest.csv \
  --check-against splits/split_manifest.csv
```

The script uses `GroupShuffleSplit`:

- Seed 42 reserves 20% of genes for test.
- Seed 43 reserves 25% of the remaining genes for validation.
- Genes and transcript-position sites cannot cross partitions.

The proportions apply to genes rather than exact site counts. The supplied manifest was verified to be reproducible from the course labels.

Data1 gene mappings are obtained from data0 where needed. The 15,137 data1 sites without gene mappings are excluded from this evaluation manifest. Data2 is not involved in splitting.

Although the manifest contains both data0 and data1, **training uses only its data0 train/validation rows**. The split originally supported a shared RF/XGBoost comparison; no RF code is needed to generate or use it.

## 6. Train XGBoost

Training requires data0 JSON, binary labels containing `gene_id,transcript_id,transcript_position,label`, and a matching manifest.

```bash
python train.py \
  --data ../data0/dataset0.json.gz \
  --labels ../data0/data.info.labelled \
  --manifest splits/split_manifest.csv \
  --output-dir artifacts/retrained
```

The output directory must be empty or absent. Outputs are:

- `selected_xgb.joblib`
- `training_report.json`
- `validation_metrics.csv`

Four candidates vary tree depth (`3` or `5`) and positive-class weight (`1` or `4`). Early stopping uses validation data. The average of validation ROC-AUC and trapezoidal PR-AUC selects the configuration.

The selected configuration is refitted on data0 train+validation. Test observations are excluded from fitting and early stopping.

| Partition | Sites |
|---|---:|
| data0 train | 73,233 |
| data0 validation | 24,773 |
| Final fitting: train + validation | 98,006 |
| data0 test | 23,832 |
| data1 test | 15,187 |

The supplied model uses depth `5`, positive-class weight `1`, `408` boosting rounds and random seed `4262`. It is **not refitted on all data0**.

Its 110 features summarize signal distributions, log dwell time, central/flanking contrasts, read count, sequence bases and motifs. Identifiers and labels are not input features. Feature extraction is shared through `xgb_features.py`.

To use a newly trained model, pass:

```text
--model artifacts/retrained/selected_xgb.joblib
```

## 7. Held-out evaluation

```bash
python evaluate_test.py \
  --data0 ../data0/dataset0.json.gz \
  --data1 ../data1/dataset1.json.gz \
  --model artifacts/selected_xgb.joblib \
  --output-dir evaluation/recomputed
```

This loads the frozen model, verifies its saved manifest and evaluates all 39,019 test observations. It never fits a model.

Recorded results are in:

```text
evaluation/test_metrics.csv
evaluation/heldout_test_predictions.csv
```

Evaluation output directories must be empty or absent. If using newly generated predictions for benchmarking, supply their path with `benchmark_m6anet.py --xgb-predictions`.

## 8. Generate intermediate-submission predictions

```bash
python run_pipeline.py \
  --data-root .. \
  --model artifacts/selected_xgb.joblib \
  --output-dir submissions \
  --team-name Genomix_haribo
```

The script predicts all sites and validates the three CSVs:

| File | Sites |
|---|---:|
| `Genomix_haribo_dataset0.csv` | 121,838 |
| `Genomix_haribo_dataset1.csv` | 90,810 |
| `Genomix_haribo_dataset2.csv` | 1,323 |

Use the registered team name. The wrapper does not retrain or upload anything. Existing output files are refused.

Full data0 predictions include training observations and must not be treated as independent test performance.

## 9. Official m6Anet benchmark

### Install the separate environment

```bash
conda create -n m6anet38 \
  --override-channels \
  -c pytorch -c conda-forge -c bioconda -c defaults \
  python=3.8 m6anet=2.0.2 cpuonly "numpy<1.24" -y

conda activate m6anet38
python -c "import torch; print(torch.__version__)"
m6anet inference --help
```

The tested environment uses PyTorch `1.6.0`. Do not install these legacy dependencies into `xgb`.

### Prepare inputs

```bash
conda activate xgb

python prepare_m6anet.py \
  --data0 ../data0/dataset0.json.gz \
  --data1 ../data1/dataset1.json.gz \
  --output-dir evaluation/m6anet_inputs

python prepare_m6anet_data2.py
```

Data0/data1 preparation selects held-out test sites. Data2 preparation selects all raw sites.

These scripts append synthetic per-site read IDs, rebuild JSON byte offsets and retain sites with at least 20 reads. All final evaluation sites met that coverage threshold.

### Run inference

```bash
conda activate m6anet38

m6anet inference \
  --input_dir evaluation/m6anet_inputs/data0 \
  --out_dir evaluation/m6anet_outputs/data0 \
  --n_processes 1 --num_iterations 1000 --seed 42 \
  --save_per_batch 100000

m6anet inference \
  --input_dir evaluation/m6anet_inputs/data1 \
  --out_dir evaluation/m6anet_outputs/data1 \
  --n_processes 1 --num_iterations 1000 --seed 42 \
  --save_per_batch 100000

m6anet inference \
  --input_dir evaluation/m6anet_inputs/data2 \
  --out_dir evaluation/m6anet_outputs/data2 \
  --n_processes 1 --num_iterations 1000 --seed 42 \
  --save_per_batch 100000
```

These commands use the package’s default HCT116 pretrained model and normalization.

**m6Anet 2.0.2 saving workaround:** the installed code uses `if (it + 1) % args.save_per_batch` and lacks a final buffer flush. The default value `2` omitted the final 8 data0 and 3 data1 sites. Setting `100000`, larger than the number of batches in these datasets, makes every batch save and restores complete coverage. Setting `1` prevents saving in this version. This workaround changes saving behavior, not weights.

### Compare data0/data1

```bash
conda activate xgb

python benchmark_m6anet.py \
  --m6anet-data0 evaluation/m6anet_outputs/data0/data.site_proba.csv \
  --m6anet-data1 evaluation/m6anet_outputs/data1/data.site_proba.csv \
  --output-dir evaluation/m6anet_recomputed
```

Both methods are scored on identical test sites. Final recorded results in `evaluation/m6anet_comparison/` cover all 39,019 original test observations.

### Compare data2 separately

Generate the XGBoost data2 CSV using section 8 first, then run:

```bash
python evaluate_data2.py \
  --labels ../data2/data.info \
  --predictions submissions/Genomix_haribo_dataset2.csv \
  --m6anet-predictions evaluation/m6anet_outputs/data2/data.site_proba.csv \
  --output-dir evaluation/data2_recomputed
```

For evaluation, data2 labels are converted using:

```text
label >= 0.5 → 1
label < 0.5 → 0
```

Original fractional labels are preserved. There are 189 exact `0.5` labels, assigned to class 1 under the agreed project convention. Data2 labels are not used for training or tuning, and prediction scores remain probabilities.

The final comparison covers all 1,323 data2 sites. Its positive rate is 71.43%, so its PR metrics are interpreted separately from data0/data1.

## 10. Results and limitations

| Dataset | Model | ROC-AUC | PR-AUC | AP |
|---|---|---:|---:|---:|
| data0+data1 test | XGBoost | 0.876746 | 0.462438 | 0.462871 |
| data0+data1 test | m6Anet | 0.878555 | 0.446030 | 0.446313 |
| data0 test | XGBoost | 0.926827 | 0.530420 | 0.531173 |
| data0 test | m6Anet | 0.935117 | 0.527973 | 0.528455 |
| data1 test | XGBoost | 0.820503 | 0.399059 | 0.399623 |
| data1 test | m6Anet | 0.813147 | 0.369707 | 0.370311 |
| data2 | XGBoost | 0.704471 | 0.860361 | 0.860479 |
| data2 | m6Anet | 0.673856 | 0.864178 | 0.864272 |

PR-AUC uses trapezoidal integration; AP uses `average_precision_score`.

XGBoost aggregates read signals into site features. m6Anet uses neural multiple-instance learning and sampling-based site pooling. Pretrained m6Anet training-data overlap is unknown.

No significance testing was performed. Test comparisons informed method choice, so they should be described as exploratory rather than independent post-selection evaluation. See `RESULTS.md` for details.

A suitable simple-baseline comparison still needs to be included in the final report.

## 11. Script reference

| Script | Purpose |
|---|---|
| `prepare_split.py` | Generate gene-level partitions |
| `train.py` | Train and save XGBoost |
| `predict.py` | Predict new JSON data |
| `xgb_features.py` | Shared feature extraction |
| `test_pipeline.py` | Run reproducibility and contract tests |
| `validate_submission.py` | Validate prediction CSVs |
| `run_pipeline.py` | Generate three full-dataset predictions |
| `evaluate_test.py` | Evaluate held-out data0/data1 sites |
| `prepare_m6anet.py` | Prepare data0/data1 test inputs |
| `prepare_m6anet_data2.py` | Prepare all data2 inputs |
| `benchmark_m6anet.py` | Compare data0/data1 predictions |
| `evaluate_data2.py` | Evaluate data2 and optional m6Anet comparison |

The repository includes the model, manifest, small example, metrics and compact site-level m6Anet scores. Full datasets, prepared signal JSON, individual-read predictions, environments, SSH keys and caches are excluded.

## 12. Working on AWS

Code can be edited locally and uploaded again, or edited remotely using VS Code Remote SSH. Local and server copies do not synchronize automatically.

After connecting to the Ubuntu server, enter the project directory:

```bash
cd ~/Group_Project/Task1
```

If feature extraction or training logic changes, retrain and update the model and reference results together.

## References

- [Official m6Anet repository](https://github.com/GoekeLab/m6anet)
- [m6Anet Bioconda package](https://bioconda.github.io/recipes/m6anet/README.html)
- Hendra et al. (2022), [Detection of m6A from direct RNA sequencing using a multiple instance learning framework](https://doi.org/10.1038/s41592-022-01666-1).
- [Miniforge](https://github.com/conda-forge/miniforge)
