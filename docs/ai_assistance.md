# AI assistance record

Codex assisted with implementation and documentation following the user's experiment requests. This provenance record does not replace the student's own interpretation or the course report.

| Request | Assistance used |
|---|---|
| Build and explain a data0 baseline | Feature extraction, scaler/logistic pipeline, gene split, prediction script, tests and workflow explanation |
| Test on data1 and document results | Frozen evaluation, overlap/subset analysis and report |
| Train data0/data1 with cross-fold validation | Conservative gene groups, grouped validation, C selection and report |
| Include data2 using option 3 labels | Inclusive 0.5 cutoff, raw-fraction preservation, position grouping and per-source evaluation |
| Prepare for GitHub | Portable paths, model/reference package, pinned runtime, example, CI and checks |

The user chose the data2 label rule. Hyperparameters were selected from training-fold average precision. Results came from executing code and were checked against predictions and saved model statistics. Annotation sources and hashes are recorded under `reference/` and `results/`.

A draft metrics export failed when NumPy integers were passed to the JSON serializer. Counts were converted to ordinary Python integers and the export was rerun and checked. Inspection of the SG-NEx annotation found headers inconsistent with the first data columns; the parser explicitly identifies ENST/ENSG prefixes.

The student should review implementation, results and source attribution before submission. Hidden leaderboard performance is unknown; data2 independence is established only for repeated source positions.
