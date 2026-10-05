"""Fixed grouped splits and training-negative motif-normalization ablations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from features import load_features, iter_sites
from label_utils import load_label_tables, write_mapping_audit
from modeling import MODELS, build_model, model_columns, evaluation_rows
from normalization import Reservoir, finish_reference, normalized_site

KEYS = ["transcript_id", "transcript_position"]


def log(message):
    print(message, flush=True)


def fingerprint(json_paths, label_paths, gene_map, split_by, gene_gtf=None,
                strip_id_version=False, unmapped_policy="error"):
    paths = json_paths + label_paths + ([gene_map] if gene_map else []) + ([gene_gtf] if gene_gtf else [])
    return {"files": [{"path": str(Path(p).resolve()), "size": Path(p).stat().st_size,
                       "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths],
            "split_by": split_by, "test_seed": 42, "validation_seed": 43,
            "feature_version": 2, "gene_gtf": gene_gtf,
            "strip_id_version": strip_id_version, "unmapped_policy": unmapped_policy}


def load_data(json_paths, label_tables, split_by="gene", unmapped_policy="error"):
    frames, stats = [], []
    for source, (path, labels) in enumerate(zip(json_paths, label_tables)):
        if labels[KEYS].isna().any().any() or labels.duplicated(KEYS).any():
            raise ValueError("Missing or duplicate site coordinates within a label file")
        if labels.label.isna().any() or not set(labels.label.unique()).issubset({0, 1}):
            raise ValueError("Only binary labels 0/1 are supported")
        log(f"Extracting original features: dataset {source}")
        sites = load_features(path)
        frame = sites.merge(labels[KEYS + ["gene_id", "label"]], on=KEYS,
                            how="inner", validate="one_to_one")
        if frame.empty:
            raise ValueError(f"No matching sites: {path}")
        excluded = int(frame.gene_id.isna().sum()) if split_by == "gene" else 0
        if excluded and unmapped_policy == "error":
            raise ValueError("Missing gene mapping: inspect unmapped_sites.csv; supply matching GTF/CSV")
        if excluded:
            frame = frame.loc[frame.gene_id.notna()].copy()
        if frame.empty:
            raise ValueError(f"No mapped labelled sites remain after filtering: {path}")
        frame["source_id"] = source
        frames.append(frame)
        stats.append({"source_id": source, "json_sites": len(sites),
                      "matched_sites": len(frame), "unmatched_sites": len(sites)-len(frame)-excluded,
                      "excluded_unmapped_matched_sites": excluded})
        log(str(stats[-1]))
    return pd.concat(frames, ignore_index=True), stats


def split_data(data, group_column, out, signature):
    manifest_path = out / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Inputs or split mode changed. Use a NEW --out-dir.")
        group_to_split = {g: part for part, groups in manifest["groups"].items() for g in groups}
        data["split"] = data[group_column].map(group_to_split)
        if data.split.isna().any():
            raise ValueError("Stored split does not cover every group")
        return data, manifest
    trainval_idx, test_idx = next(GroupShuffleSplit(test_size=.2, random_state=42).split(
        data, groups=data[group_column]))
    trainval = data.iloc[trainval_idx]
    train_sub, val_sub = next(GroupShuffleSplit(test_size=.25, random_state=43).split(
        trainval, groups=trainval[group_column]))
    data["split"] = "test"
    data.loc[trainval.index[train_sub], "split"] = "train"
    data.loc[trainval.index[val_sub], "split"] = "validation"
    manifest = {"signature": signature, "group_column": group_column,
                "groups": {part: sorted(data.loc[data.split.eq(part), group_column].unique().tolist())
                           for part in ("train", "validation", "test")}}
    group_sets = [set(x) for x in manifest["groups"].values()]
    assert all(not group_sets[i] & group_sets[j] for i in range(3) for j in range(i))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return data, manifest


def learn_reference(paths, data):
    rng = np.random.default_rng(2026)
    reservoirs, counts = {}, Counter()
    global_reservoir = Reservoir(20000, rng)
    for source, path in enumerate(paths):
        negative = data.loc[(data.source_id == source) & data.split.eq("train") & data.label.eq(0)]
        keys = set(zip(negative.transcript_id, negative.transcript_position))
        log(f"Learning reference: dataset {source}; {len(keys)} training-negative sites")
        for transcript, pos, seq, reads in iter_sites(path):
            if (transcript, pos) not in keys:
                continue
            arr = np.asarray(reads, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 9 or not len(arr) or not np.isfinite(arr).all():
                raise ValueError("Invalid training-negative read matrix")
            # Equal capped per-site contribution limits the influence of high coverage.
            if len(arr) > 20:
                arr = arr[rng.choice(len(arr), 20, replace=False)]
            motif = seq[1:6]
            if motif not in reservoirs:
                reservoirs[motif] = Reservoir(20000, rng)
            reservoirs[motif].add(arr)
            global_reservoir.add(arr)
            counts[motif] += 1
    return finish_reference(reservoirs, global_reservoir, counts, min_sites=20)


def add_normalized(paths, data, reference):
    rows = []
    for source, path in enumerate(paths):
        subset = data.loc[data.source_id.eq(source)]
        keys = set(zip(subset.transcript_id, subset.transcript_position))
        log(f"Extracting normalized read-distribution features: dataset {source}")
        processed = 0
        for transcript, pos, seq, reads in iter_sites(path):
            if (transcript, pos) in keys:
                rows.append({"source_id": source, "transcript_id": transcript,
                             "transcript_position": pos, **normalized_site(reads, seq, reference)})
                processed += 1
                if processed % 20000 == 0:
                    log(f"Normalized features: dataset {source}, {processed} sites")
    return data.merge(pd.DataFrame(rows), on=["source_id", *KEYS],
                      how="left", validate="one_to_one")


def write_split_audit(data, group_column, out):
    if data[group_column].isna().any() or data.split.isna().any():
        raise ValueError("Missing groups or split assignments")
    for column in (group_column, "transcript_id"):
        if data.groupby(column).split.nunique().gt(1).any():
            raise ValueError(f"Leakage: {column} occurs in multiple splits")
    if data.duplicated(["source_id", *KEYS]).any():
        raise ValueError("Duplicate observations within a dataset")
    rows = []
    for part in ("train", "validation", "test"):
        subset = data.loc[data.split.eq(part)]
        for source in ["combined", *sorted(data.source_id.unique())]:
            block = subset if source == "combined" else subset.loc[subset.source_id.eq(source)]
            rows.append({"split": part, "source_id": source, "sites": len(block),
                         "genes": block.gene_id.nunique(), "transcripts": block.transcript_id.nunique(),
                         "positives": int(block.label.sum()),
                         "positive_fraction": float(block.label.mean()) if len(block) else None})
    pd.DataFrame(rows).to_csv(out / "split_audit.csv", index=False)
    repeated = data.loc[data.duplicated(KEYS, keep=False)]
    discordant = repeated.groupby(KEYS).label.nunique().gt(1).sum()
    (out / "split_checks.json").write_text(json.dumps({
        "group_column": group_column, "group_overlap": 0, "transcript_overlap": 0,
        "cross_dataset_repeated_coordinates": len(repeated.drop_duplicates(KEYS)),
        "cross_dataset_discordant_coordinates": int(discordant),
        "observation_key": ["source_id", *KEYS],
        "duplicate_policy": "retain independent observations across datasets"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--gene-map")
    parser.add_argument("--gene-gtf", help="GTF matching the dataset reference release; .gz supported")
    parser.add_argument("--strip-id-version", action="store_true",
                        help="Strip terminal numeric ID versions for mapping only")
    parser.add_argument("--unmapped-policy", choices=["error", "drop"], default="error")
    parser.add_argument("--split-by", choices=["gene", "transcript"], default="gene")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if len(args.json) != len(args.labels):
        parser.error("Each JSON needs one label file, paired in order")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "test_metrics.csv").exists():
        raise ValueError("This experiment already evaluated test data. Do not tune it further.")
    label_tables = load_label_tables(args.labels, args.gene_map, args.gene_gtf, args.strip_id_version)
    mapping_stats = write_mapping_audit(label_tables, out)
    for table in label_tables:
        if table.label.isna().any() or not set(table.label.unique()).issubset({0, 1}):
            raise ValueError("Only binary labels 0/1 supported; do not threshold unresolved data2 labels")
    if args.split_by == "gene" and args.unmapped_policy == "error" and any(
            row["unmapped_sites"] for row in mapping_stats):
        raise ValueError("Unmapped label sites remain. Inspect unmapped_sites.csv; supply --gene-gtf/--gene-map. "
                         "Explicit --unmapped-policy drop excludes them and changes the evaluated population.")
    signature = fingerprint(args.json, args.labels, args.gene_map, args.split_by,
                            args.gene_gtf, args.strip_id_version, args.unmapped_policy)
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache = out / f"original_features_{cache_key}.joblib"
    if cache.exists():
        data, stats = joblib.load(cache)
        log("Loaded cached original features")
    else:
        data, stats = load_data(args.json, label_tables, args.split_by, args.unmapped_policy)
        joblib.dump((data, stats), cache, compress=3)
    group_column = "gene_id" if args.split_by == "gene" else "transcript_id"
    if data[group_column].isna().any():
        raise ValueError("Missing gene mapping. Supply --gene-map or --gene-gtf.")
    if args.split_by == "transcript":
        log("TRANSCRIPT split: gene-level leakage between different isoforms remains possible.")
    data, manifest = split_data(data, group_column, out, signature)
    write_split_audit(data, group_column, out)
    for part in ("train", "validation", "test"):
        subset = data.loc[data.split.eq(part)]
        if subset.label.nunique() != 2:
            raise ValueError(f"{part} must contain both classes")
        log(f"{part}: {len(subset)} observations; {subset[group_column].nunique()} groups")
    data[["source_id", *KEYS, "gene_id", "split"]].to_csv(out / "split_membership.csv", index=False)
    augmented_cache = out / f"augmented_features_{cache_key}.joblib"
    reference_path = out / "motif_reference.joblib"
    if augmented_cache.exists() and reference_path.exists():
        data = joblib.load(augmented_cache)
        reference = joblib.load(reference_path)
        log("Loaded cached normalized features")
    else:
        reference = learn_reference(args.json, data)
        joblib.dump(reference, reference_path)
        data = add_normalized(args.json, data, reference)
        joblib.dump(data, augmented_cache, compress=3)
    train = data.loc[data.split.eq("train")]
    val = data.loc[data.split.eq("validation")]
    pd.DataFrame([{"motif": m, "training_negative_sites": n,
                   "uses_global_fallback": m not in reference["motifs"]}
                  for m, n in sorted(reference["negative_site_counts"].items())]).to_csv(
                      out / "motif_reference_summary.csv", index=False)
    rows = []
    for name in MODELS:
        columns = model_columns(data, name)
        pipeline = build_model(columns, name)
        log(f"Training {name}: {len(columns)} input columns")
        pipeline.fit(train[columns], train.label)
        scores = pipeline.predict_proba(val[columns])[:, 1]
        model_rows = evaluation_rows(val, scores, name)
        rows.extend(model_rows)
        bundle = {"pipeline": pipeline, "columns": columns, "name": name,
                  "reference": reference if name in MODELS[2:] else None,
                  "signature": signature}
        joblib.dump(bundle, out / f"{name}.joblib", compress=3)
        log(json.dumps(model_rows[0]))
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation_metrics.csv", index=False)
    winner = table.loc[table.dataset.eq("combined")].sort_values(
        "pr_auc", ascending=False, kind="stable").iloc[0].model
    summary = {"selected_model": winner, "selection_metric": "combined validation average precision",
               "split_by": args.split_by, "test_evaluated": False, "datasets": stats,
               "cache_file": augmented_cache.name, "signature": signature,
               "counts": {p: int(data.split.eq(p).sum()) for p in ("train", "validation", "test")},
               "unique_sites": len(data.drop_duplicates(KEYS)),
               "repeated_site_observations": int(data.duplicated(KEYS).sum()),
               "unmapped_gene_sites": int(data.gene_id.isna().sum()),
               "mapping_coverage": mapping_stats, "metric_definition": "site-level ROC-AUC and average precision (AP)"}
    (out / "experiment.json").write_text(json.dumps(summary, indent=2))
    log(f"Selected by validation: {winner}. Test scores have NOT been calculated.")


if __name__ == "__main__":
    main()
