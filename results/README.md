# Recorded results

These summaries and audits describe executed local experiments, not CI-generated results.

| Folder | Experiment |
|---|---|
| `data0/` | Original gene-separated evaluation followed by full-data0 refit |
| `data1_evaluation/` | Frozen data0 models evaluated on data1 |
| `data0_data1/` | Data0 + data1 grouped training and five-fold validation |
| `all_datasets/` | Current three-dataset experiment; inclusive data2 cutoff 0.5 |

Each folder contains `metrics.json` and the original `verification.json`. The data2 preparation audit is also included. `repository_verification.json` records publication checks separately so historical test counts keep their original meaning.

Absolute paths were made relative. Historic `reference/raw/` paths document omitted annotation files; their URLs/checksums are in `reference/provenance.json`. Distributed model hashes are in `models/manifest.json`.

Row-level splits, feature caches and large prediction CSVs stay under ignored `artifacts/` and can be regenerated with the original data. Changing group membership prevents a controlled comparison across runs.
