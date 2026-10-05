# Gene-disjoint m6A classifier experiments

Task 1 predicts **transcript-site** scores. The grouping unit is **gene**, while
ROC-AUC and average precision are still calculated over individual site
observations. This update retains the existing feature definitions, HGB settings,
training-negative normalization and four-model ablation.

## Important: existing results are historical

`results/` and `trained_models/` remain the original **transcript-disjoint**
experiment. They have not been retrained with gene splitting. Its experiment.json
records 15,137 observations without gene mapping out of 212,648 matched
observations. New gene-disjoint AUC values require the original JSON and label
files plus sufficient mapping. Those input datasets are not included in this ZIP.
Use a **new output directory**; do not mix old models, caches or manifests into it.

## 1. Check gene mapping before scanning JSON

```bash
python -m pip install -r requirements.txt
python prepare_gene_mapping.py \
  --labels data/data.info.labelled data1/data.info.labelled \
  --out-dir runs/gene_mapping_check
```

The script pools all transcript-to-gene pairs from every supplied label table,
then maps all datasets through this shared lookup. Thus dataset0's gene IDs
also map shared transcripts in dataset1. No `gene_id` is used as a model feature.
The preflight uses labels only and does not train a model.

Outputs:

- `gene_mapping_coverage.csv`: mapping coverage by dataset.
- `resolved_gene_map.csv`: deduplicated transcript-to-gene pairs.
- `unmapped_sites.csv`: dataset-specific sites that still lack a gene.

If any label sites remain unmapped, the preflight exits unsuccessfully **after
writing these reports**. Complete the map using the GTF matching the original
reference assembly and annotation release, or a verified CSV:

```bash
python prepare_gene_mapping.py \
  --labels data/data.info.labelled data1/data.info.labelled \
  --gene-gtf reference/matching_annotation.gtf.gz \
  --out-dir runs/gene_mapping_check
```

`--gene-map transcript_gene.csv` accepts columns `transcript_id,gene_id`.
CSV and GTF can be supplied together; conflicting gene assignments are errors.
GTF plain text and gzip are supported. The package cannot determine the correct
annotation release from the supplied experiment archive; verify it upstream.

ID matching is exact by default. If annotation and dataset differ only by
terminal numeric versions, add `--strip-id-version` to both preflight and
training. It removes `.N` from transcript/gene IDs for mapping and grouping,
but **retains the original transcript ID** when joining JSON and emitting
predictions. Conflicting canonical aliases remain errors. Inspect coverage and
reference compatibility before enabling this option.

## 2. Train using one shared gene split across datasets

```bash
python train_experiments.py \
  --json data/dataset0.json.gz data1/dataset1.json.gz \
  --labels data/data.info.labelled data1/data.info.labelled \
  --gene-gtf reference/matching_annotation.gtf.gz \
  --split-by gene \
  --out-dir runs/gene01
```

If dataset0 already covers every transcript, omit `--gene-gtf`. If needed, add
`--gene-map` or `--strip-id-version` as above. The default split is now `gene`.
The JSON and label paths are paired by order; source_id 0 and 1 refer to those
positions, not inferred biological dataset names.

All datasets are concatenated **before** GroupShuffleSplit. Test takes 20% of
genes with seed 42; validation takes 25% of remaining genes with seed 43,
leaving approximately 60/20/20 genes. Site counts may differ from those ratios.
This is a grouped random split, not class-stratified. Each combined partition
must contain both classes. Per-dataset single-class partitions yield undefined
AUC (empty metric values), not a fabricated score.

Missing mapping fails by default before expensive feature extraction. There is
no fallback that assigns every missing transcript to one `unknown` gene or
silently switches to transcript splitting. Explicit `--unmapped-policy drop`
excludes unmapped matched observations from all partitions; label coverage and
excluded matched counts are reported. This changes the evaluation population;
prefer completing the map. Do not interpret the resulting AUC as covering all
original sites. Sites lacking labels are excluded by the existing inner join,
with unmatched JSON counts reported.

The observation key is `(source_id, transcript_id, transcript_position)`.
Identical coordinates from different cell lines retain their independent labels
and features, even when labels disagree. They must share one gene partition.
Within-dataset duplicate coordinates are errors, never silently deduplicated.

New audit outputs:

- `split_manifest.json`: fixed assignments and input/configuration signature.
- `split_membership.csv`: source, transcript, position, gene, partition.
- `split_audit.csv`: site/gene/transcript counts, positives and prevalence for
  each partition, combined and per dataset.
- `split_checks.json`: verified zero grouping/transcript overlap across splits,
  cross-dataset coordinate repetition and label disagreement counts.

The saved manifest is reused for identical inputs. Changes to input paths,
sizes, mtimes or mapping options require a fresh run directory. The signature
is a cache guard, not a cryptographic content checksum. Do not choose seeds or
new runs by looking for higher test scores.

## 3. Models and training-only preprocessing

| Model | Features |
| --- | --- |
| logistic_simple | Nine signal means and sequence context |
| hgb_original | Original signal summaries, sequence context, read count |
| hgb_normalized | Original features plus motif-reference normalized summaries |
| hgb_distribution | Above plus normalized read tails and joint deviation |

All methods use exactly the same split. HGB remains max_iter=100,
max_leaf_nodes=15, l2_regularization=1, random_state=42, early_stopping=False.
Gene/source/transcript identifiers are absent from the feature columns.

Imputation, scaling, encoding and classifiers fit only train. Motif normalization
uses only training-negative sites: at most 20 sampled reads per site, 20,000
reservoir reads per motif, median and 1.4826*MAD, with pooled fallback for fewer
than 20 negative sites. These are empirical negative references, not proof that
all reads are unmodified. Distribution tails at z=+/-2 and joint deviations
are signal features, not modification probabilities or modification ratios.

Training writes cached features and model bundles to the run directory. JSON
is scanned for original features, reference learning and normalized features;
subsequent identical runs can reuse caches. Reserve disk space for them.

`validation_metrics.csv` reports combined and per-dataset metrics. Model selection
uses combined validation AP. Combined evaluation weights each cell-line site
observation once, including repeated coordinates; inspect per-dataset results.
`pr_auc` means **sklearn average_precision_score (AP)**, not trapezoidal PR area.
Scores remain continuous. Metrics include site count, positives and prevalence.
The official course metric, if different, must be implemented consistently for
all benchmark methods. Non-binary labels are rejected; unresolved dataset2
values such as 0.95/0.75/0.25 must not be thresholded without clarification.

## 4. Final test and inference

After model selection, run once:

```bash
python evaluate_test.py --run-dir runs/gene01
python analyze_validation.py --run-dir runs/gene01
```

Test evaluates Logistic, fixed original HGB and the validation-selected model.
It emits test_metrics.csv and source-specific test predictions. The existing
one-time test guard prevents further training/evaluation in that directory.
An unseen-gene test is an internal evaluation design, not a claim about the
professor's leaderboard composition. A new gene split also does not erase any
previous influence from inspecting related data/results.

For course submissions, use **the same chosen model** to produce three files:

```bash
python predict.py --json data/dataset0.json.gz --model runs/gene01/hgb_original.joblib --output dataset0_scores.csv
python predict.py --json data1/dataset1.json.gz --model runs/gene01/hgb_original.joblib --output dataset1_scores.csv
python predict.py --json data2/dataset2.json.gz --model runs/gene01/hgb_original.joblib --output dataset2_scores.csv
```

Replace the model path consistently if choosing another saved model. Predictions
need no labels or gene IDs and have `transcript_id,transcript_position,score`.
Scores on sites used for fitting are not independent test performance.
Archive models used sklearn 1.8.0; retrain locally when versions differ. Load
joblib bundles only from trusted sources.

## 5. Follow-up evaluation designs

These are separate protocols; the gene split script implements the first only.

| Design | Training and selection | Final evaluation |
| --- | --- | --- |
| Gene-disjoint | Shared train genes across data0/data1; tune on validation genes | Held-out genes, metrics separately per cell line |
| Cross-cell-line | Train data0; tune using a held-out data0 partition only | All eligible data1 sites, report gene overlap |
| Cross-cell-line plus unseen genes | Train/tune data0 | Restrict data1 to genes absent from all fitted/selection data |

Do not call the current joint data0/data1 gene run a cross-cell-line experiment:
it includes both cell lines in training. A source holdout needs an independently
implemented assignment protocol, training-only references, and its own manifest.
Do not reuse the jointly trained bundles for that claim.

For a same-condition m6Anet benchmark, freeze the eligible test membership from
split_membership.csv. For each dataset separately, align HGB predictions,
labels and m6Anet site-level `probability_modified` by exact transcript/position
coordinates. Reject within-source duplicates. Report total eligible sites,
predicted sites, missing sites, and the common-site intersection. Compare both
methods with this package's metrics on identical common sites; also report
full available coverage separately. Never impute missing predictions as zero.
Record m6Anet version, weights, training provenance, minimum-read filter, seed,
inference configuration, and possible overlap of pretrained training data.
Using official pretrained weights compares available tools; retraining on the
same train genes compares learned methods. Individual-read probabilities and
`mod_ratio` are not substitutes for the site probability. This package does
not run m6Anet or contain its predictions, and no paper AUC is directly comparable.

## Verification

```bash
python -m unittest -v test_gene_split.py
```

Checks cover shared mappings, conflicting mappings, GTF version aliases,
unknown transcripts, duplicate sites, opposite cross-cell-line labels,
gene/isoform consistency across datasets and saved-manifest reuse.
