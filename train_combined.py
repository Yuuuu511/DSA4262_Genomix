"""Combine labelled datasets, tune logistic regression with grouped CV, test once."""
import argparse
import csv
import gzip
import json
import platform
import re
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import load_metadata
from features import FEATURE_NAMES, FEATURE_VERSION, iter_sites
from train import digest, metrics, read_labels

ANNOTATION_URL = "https://sg-nex-data.s3.amazonaws.com/data/annotations/txt_file/Homo_sapiens.GRCh38.91.annotations-transcripts.txt"
PATCH_URL = "https://ftp.ensembl.org/pub/release-96/gtf/homo_sapiens/Homo_sapiens.GRCh38.96.chr_patch_hapl_scaff.gtf.gz"
PROJECT_DIR = Path(__file__).resolve().parent


class GeneGroups:
    """Conservative union of gene identifiers linked by transcript or gene name."""
    def __init__(self):
        self.parent = {}

    def find(self, gene):
        self.parent.setdefault(gene, gene)
        if self.parent[gene] != gene:
            self.parent[gene] = self.find(self.parent[gene])
        return self.parent[gene]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def build_gene_map(reference, metadata, annotation_path, lookup_path, patch_path):
    supplied = {}
    for (transcript, _), (gene, _) in reference.items():
        if transcript in supplied and supplied[transcript] != gene:
            raise ValueError(f"Conflicting supplied gene IDs for {transcript}")
        supplied[transcript] = gene
    wanted = set(supplied) | {t for t, _ in metadata}
    candidates = {t: {g} for t, g in supplied.items()}
    preferred = dict(supplied)
    sources = {t: "data0_labels" for t in supplied}
    compared, conflicts = 0, []
    # The downloaded table has an extra data column and swapped first-column headers.
    # Identify the transcript/gene columns by stable-ID prefixes, not the header text.
    with open(annotation_path) as handle:
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            if len(cells) < 2 or not cells[0].startswith("ENST") or not cells[1].startswith("ENSG"):
                continue
            transcript, gene = cells[0].split(".")[0], cells[1].split(".")[0]
            if transcript not in wanted:
                continue
            candidates.setdefault(transcript, set()).add(gene)
            if transcript in supplied:
                compared += 1
                if supplied[transcript] != gene:
                    conflicts.append(transcript)
            if transcript not in preferred:
                preferred[transcript], sources[transcript] = gene, "SGNEx_Ensembl91"
    lookup = json.loads(Path(lookup_path).read_text())
    for transcript, record in lookup.items():
        if transcript not in wanted or record is None:
            continue
        gene = record.get("Parent", "")
        if not gene.startswith("ENSG") or record.get("object_type") != "Transcript":
            raise ValueError(f"Invalid Ensembl lookup for {transcript}")
        gene = gene.split(".")[0]
        candidates.setdefault(transcript, set()).add(gene)
        if transcript not in preferred:
            preferred[transcript], sources[transcript] = gene, "Ensembl_REST_snapshot"
    names = defaultdict(set)
    with gzip.open(patch_path, "rt") as handle:
        for line in handle:
            if "\ttranscript\t" not in line:
                continue
            gene = re.search(r'gene_id "([^"]+)"', line)[1]
            transcript = re.search(r'transcript_id "([^"]+)"', line)[1]
            name = re.search(r'gene_name "([^"]+)"', line)
            if name:
                names[name[1]].add(gene)
            if transcript in wanted:
                candidates.setdefault(transcript, set()).add(gene)
                if transcript not in preferred:
                    preferred[transcript], sources[transcript] = gene, "Ensembl96_patch_GTF"
    missing = wanted - set(preferred)
    if missing:
        raise ValueError(f"Unmapped transcripts; refusing an unsafe split: {sorted(missing)}")
    groups = GeneGroups()
    for genes in candidates.values():
        first = min(genes)
        for gene in genes:
            groups.union(first, gene)
    relevant = set(groups.parent)
    # Alternate patch gene identifiers with the same annotation gene name remain together.
    for genes in names.values():
        if genes & relevant:
            first = min(genes)
            for gene in genes:
                groups.union(first, gene)
    group_by_transcript = {t: groups.find(preferred[t]) for t in wanted}
    info = {"transcripts": len(wanted), "unmapped_transcripts": 0,
            "preferred_gene_source_counts": dict(Counter(sources.values())),
            "SGNEx_vs_supplied_transcripts_compared": compared,
            "SGNEx_vs_supplied_gene_conflicts": len(conflicts),
            "group_rule": "union gene IDs sharing transcript mappings or Ensembl96 gene names",
            "preferred_gene_ids": len(set(preferred.values())),
            "conservative_gene_groups": len(set(group_by_transcript.values()))}
    return preferred, group_by_transcript, sources, info


def load_gene_map(reference, metadata, mapping_path, provenance_path, labels0, labels1):
    """Load the portable mapping snapshot, bound to its original label files."""
    provenance = json.loads(Path(provenance_path).read_text())
    if digest(mapping_path) != provenance["mapping_sha256"]:
        raise ValueError("Gene mapping does not match its provenance checksum")
    for key, path in (("labels0", labels0), ("labels1", labels1)):
        if digest(path) != provenance["label_sources"][key]["sha256"]:
            raise ValueError(f"{key} differs from the mapping snapshot; rebuild the gene map with raw references")
    genes, groups, sources, group_by_gene = {}, {}, {}, {}
    with open(mapping_path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"transcript_id", "gene_id", "gene_group", "mapping_source"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Gene mapping is missing required columns")
        for row in reader:
            transcript, gene, group = row["transcript_id"], row["gene_id"], row["gene_group"]
            if transcript in genes or not transcript.startswith("ENST") or not gene.startswith("ENSG") or not group.startswith("ENSG"):
                raise ValueError(f"Duplicate or invalid gene mapping: {transcript}")
            if gene in group_by_gene and group_by_gene[gene] != group:
                raise ValueError(f"Gene ID occurs in multiple split groups: {gene}")
            group_by_gene[gene] = group
            genes[transcript], groups[transcript], sources[transcript] = gene, group, row["mapping_source"]
    wanted = {t for t, _ in reference} | {t for t, _ in metadata}
    if set(genes) != wanted:
        raise ValueError("Gene mapping does not cover exactly the requested transcripts")
    if any(genes[t] != gene for (t, _), (gene, _) in reference.items()):
        raise ValueError("Gene mapping conflicts with data0's supplied gene IDs")
    info = provenance["gene_mapping"]
    actual = (len(genes), len(set(genes.values())), len(set(groups.values())), dict(Counter(sources.values())))
    expected = (info["transcripts"], info["preferred_gene_ids"], info["conservative_gene_groups"], info["preferred_gene_source_counts"])
    if actual != expected:
        raise ValueError("Gene mapping counts differ from its provenance")
    return genes, groups, sources, info


def extract_dataset(name, data, metadata, genes, groups):
    features, y, transcripts, positions, seen = [], [], [], [], set()
    for transcript, position, x in iter_sites(data):
        key = (transcript, position)
        if key in seen or key not in metadata:
            raise ValueError(f"{name}: duplicate signal key or missing label: {key}")
        seen.add(key)
        row = metadata[key]
        if "n_reads" in row and int(row["n_reads"]) != round(np.expm1(x[18])):
            raise ValueError(f"{name}: read count mismatch: {key}")
        features.append(x)
        y.append(int(float(row["label"])))
        transcripts.append(transcript)
        positions.append(position)
        if len(features) % 20000 == 0:
            print(f"{name}: extracted {len(features):,} sites", flush=True)
    if seen != set(metadata):
        raise ValueError(f"{name}: signal/label coordinates do not match completely")
    return {"X": np.asarray(features), "y": np.asarray(y, dtype=np.int8),
            "transcripts": np.asarray(transcripts), "positions": np.asarray(positions),
            "genes": np.asarray([genes[t] for t in transcripts]),
            "groups": np.asarray([groups[t] for t in transcripts]),
            "datasets": np.full(len(y), name)}


def threshold_labels(labels, threshold=0.5):
    values = np.asarray(labels, dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Fractional labels must be finite and within [0, 1]")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Label threshold must be within (0, 1]")
    return (values >= threshold).astype(np.int8)


def extract_data2(data, labels_path, threshold):
    metadata = {}
    with open(labels_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"transcript_id", "transcript_position", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Data2 metadata is missing required columns")
        for row in reader:
            key = (row["transcript_id"], row["transcript_position"])
            if key in metadata:
                raise ValueError(f"Duplicate data2 label key: {key}")
            threshold_labels([float(row["label"])], threshold)
            metadata[key] = row
    contexts = {}
    opener = gzip.open if Path(data).suffix == ".gz" else open
    with opener(data, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            for _, positions in json.loads(line).items():
                for position, context in positions.items():
                    if len(context) != 1:
                        raise ValueError("Data2 site must have one sequence context")
                    sequence = next(iter(context)).upper().replace("U", "T")
                    if position in contexts and sequence != contexts[position]:
                        raise ValueError("Data2 position has inconsistent sequence across conditions")
                    contexts[position] = sequence
    features, raw, transcripts, positions, seen = [], [], [], [], set()
    for transcript, position, x in iter_sites(data):
        key = (transcript, position)
        if key in seen or key not in metadata:
            raise ValueError(f"Duplicate data2 signal site or missing label: {key}")
        seen.add(key)
        row = metadata[key]
        if "n_reads" in row and int(row["n_reads"]) != round(np.expm1(x[18])):
            raise ValueError(f"Data2 read count mismatch: {key}")
        features.append(x)
        raw.append(float(row["label"]))
        transcripts.append(transcript)
        positions.append(position)
    if not seen or seen != set(metadata):
        raise ValueError("Data2 signal and label coordinates do not match completely")
    # tx_id_0..6 identify repeated conditions, not annotated biological genes.
    # Keep all conditions of a source position together, using a disjoint namespace.
    group_ids = [f"data2_position_{p}_{contexts[p]}" for p in positions]
    print(f"data2: extracted {len(raw):,} observations in {len(set(group_ids))} source-position groups", flush=True)
    return {"X": np.asarray(features), "raw_labels": np.asarray(raw),
            "y": threshold_labels(raw, threshold), "transcripts": np.asarray(transcripts),
            "positions": np.asarray(positions), "genes": np.full(len(raw), ""),
            "groups": np.asarray(group_ids), "datasets": np.full(len(raw), "data2")}


def check_separation(values, a, b):
    if set(values["groups"][a]) & set(values["groups"][b]):
        raise ValueError("Leakage between split groups")
    natural = values.get("datasets", np.full(len(values["y"]), "data0")) != "data2"
    for field in ("genes", "transcripts"):
        if set(values[field][a[natural[a]]]) & set(values[field][b[natural[b]]]):
            raise ValueError(f"Natural-data leakage between splits for {field}")
    left = set(zip(values["transcripts"][a], values["positions"][a]))
    right = set(zip(values["transcripts"][b], values["positions"][b]))
    if left & right:
        raise ValueError("Shared coordinate occurs on both sides of split")
    data2_a = set(values["positions"][a[~natural[a]]])
    data2_b = set(values["positions"][b[~natural[b]]])
    if data2_a & data2_b:
        raise ValueError("Related data2 conditions occur on both sides of split")
    for index in (a, b):
        if len(np.unique(values["y"][index])) != 2:
            raise ValueError("Every training/validation/test subset needs both classes")


def make_splits(values, seed=42, n_folds=5):
    y, groups = values["y"], values["groups"]
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, test = next(outer.split(values["X"], y, groups=groups))
    check_separation(values, train, test)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + 1)
    folds = []
    for fit_local, val_local in cv.split(values["X"][train], y[train], groups[train]):
        fit, val = train[fit_local], train[val_local]
        check_separation(values, fit, val)
        folds.append((fit, val))
    validation_rows = np.concatenate([v for _, v in folds])
    if len(validation_rows) != len(train) or set(validation_rows.tolist()) != set(train.tolist()):
        raise ValueError("Every outer-training row must be validated exactly once")
    return train, test, folds


def pipeline(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="lbfgs", max_iter=2000))


def stats(values, index):
    y = values["y"][index]
    natural = values["datasets"][index] != "data2"
    return {"observations": len(index), "positives": int(y.sum()),
            "positive_fraction": float(y.mean()),
            "unique_site_coordinates": len(set(zip(values["transcripts"][index], values["positions"][index]))),
            "gene_groups": len(np.unique(values["groups"][index][natural])),
            "split_groups": len(np.unique(values["groups"][index])),
            "data2_position_groups": len(np.unique(values["groups"][index][~natural])),
            "dataset_observations": dict(Counter(values["datasets"][index].tolist()))}


def write_predictions(path, values, index, score, fold=None):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "transcript_id", "transcript_position", "gene_id", "gene_group", "label", "score"]
                        + (["raw_label"] if "raw_labels" in values else [])
                        + (["validation_fold"] if fold is not None else []))
        for i, p in zip(index, score):
            row = [values["datasets"][i], values["transcripts"][i], values["positions"][i],
                   values["genes"][i], values["groups"][i], int(values["y"][i]), float(p)]
            writer.writerow(row + ([float(values["raw_labels"][i])] if "raw_labels" in values else [])
                            + ([int(fold[i])] if fold is not None else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data0", required=True)
    parser.add_argument("--labels0", required=True)
    parser.add_argument("--data1", required=True)
    parser.add_argument("--labels1", required=True)
    parser.add_argument("--data2")
    parser.add_argument("--labels2")
    parser.add_argument("--data2-threshold", type=float, default=0.5)
    parser.add_argument("--base-cache-dir", default="artifacts/combined")
    parser.add_argument("--out")
    parser.add_argument("--gene-map", default=str(PROJECT_DIR / "reference/transcript_gene_map.csv"))
    parser.add_argument("--gene-map-provenance", default=str(PROJECT_DIR / "reference/provenance.json"))
    parser.add_argument("--annotation", help="Raw SG-NEx annotation; supply all three raw reference arguments to rebuild mapping")
    parser.add_argument("--lookup", help="Saved Ensembl transcript lookup JSON")
    parser.add_argument("--patch-gtf", help="Ensembl 96 GTF including patch contigs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    args = parser.parse_args()
    if bool(args.data2) != bool(args.labels2):
        raise ValueError("Provide both --data2 and --labels2")
    if args.data2:
        threshold_labels([0.0, 1.0], args.data2_threshold)
    if args.out is None:
        args.out = "artifacts/all_datasets" if args.data2 else "artifacts/combined"
    if args.data2 and Path(args.out).resolve() == Path(args.base_cache_dir).resolve():
        raise ValueError("Use a separate output directory for the all-datasets model")
    if args.folds < 2 or not args.c_values or any(not np.isfinite(c) or c <= 0 for c in args.c_values):
        raise ValueError("Need at least two folds and positive C values")
    raw_references = (args.annotation, args.lookup, args.patch_gtf)
    if any(raw_references) and not all(raw_references):
        raise ValueError("To rebuild the gene map, provide --annotation, --lookup and --patch-gtf together")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference = read_labels(args.labels0)
    metadata0, metadata1 = load_metadata(args.labels0), load_metadata(args.labels1)
    if all(raw_references):
        genes, groups, map_sources, map_info = build_gene_map(reference, metadata1, *raw_references)
        reference_keys = ("annotation", "lookup", "patch_gtf")
    else:
        genes, groups, map_sources, map_info = load_gene_map(
            reference, metadata1, args.gene_map, args.gene_map_provenance, args.labels0, args.labels1)
        reference_keys = ("gene_map", "gene_map_provenance")
    print("Gene mapping:", json.dumps(map_info), flush=True)
    with (out / "transcript_gene_map.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["transcript_id", "gene_id", "gene_group", "mapping_source"])
        writer.writerows((t, genes[t], groups[t], map_sources[t]) for t in sorted(genes))
    source = {key: {"path": str(Path(getattr(args, key)).resolve()), "sha256": digest(getattr(args, key))}
              for key in ("data0", "labels0", "data1", "labels1", *reference_keys)}
    source["feature_version"] = FEATURE_VERSION
    source["feature_names"] = FEATURE_NAMES
    base_source = dict(source)
    if args.data2:
        for key in ("data2", "labels2"):
            source[key] = {"path": str(Path(getattr(args, key)).resolve()), "sha256": digest(getattr(args, key))}
        source["data2_label_rule"] = f"raw label >= {args.data2_threshold:g} is positive"
        source["data2_group_rule"] = "all conditions of the same source position stay together"
    cache, signature_path = out / "features.npz", out / "features_metadata.json"
    if cache.exists() and signature_path.exists() and json.loads(signature_path.read_text()) == source:
        with np.load(cache, allow_pickle=False) as stored:
            values = {k: stored[k] for k in stored.files}
        print("Reused combined feature cache with verified source hashes", flush=True)
    else:
        base_dir = Path(args.base_cache_dir)
        base_signature = base_dir / "features_metadata.json"
        if args.data2 and (base_dir / "features.npz").exists() and base_signature.exists() and json.loads(base_signature.read_text()) == base_source:
            with np.load(base_dir / "features.npz", allow_pickle=False) as stored:
                values = {k: stored[k] for k in stored.files}
            if any(groups[t] != g for t, g in zip(values["transcripts"], values["groups"])):
                raise ValueError("Cached natural-data groups differ from current gene mapping")
            print("Reused data0/data1 feature cache with verified source hashes", flush=True)
        else:
            a = extract_dataset("data0", args.data0, metadata0, genes, groups)
            b = extract_dataset("data1", args.data1, metadata1, genes, groups)
            values = {k: np.concatenate([a[k], b[k]]) for k in a}
        if args.data2:
            values["raw_labels"] = values["y"].astype(float)
            c = extract_data2(args.data2, args.labels2, args.data2_threshold)
            values = {k: np.concatenate([values[k], c[k]]) for k in values}
        np.savez_compressed(cache, **values)
        signature_path.write_text(json.dumps(source, indent=2) + "\n")
    train, test, folds = make_splits(values, args.seed, args.folds)
    n = len(values["y"])
    fold_ids = np.zeros(n, dtype=int)
    for fold, (_, val) in enumerate(folds, 1):
        fold_ids[val] = fold
    with (out / "split.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "transcript_id", "transcript_position", "gene_id", "gene_group", "label", "split", "validation_fold"]
                        + (["raw_label"] if "raw_labels" in values else []))
        for i in range(n):
            writer.writerow([values["datasets"][i], values["transcripts"][i], values["positions"][i],
                             values["genes"][i], values["groups"][i], int(values["y"][i]),
                             "train" if fold_ids[i] else "test", int(fold_ids[i])]
                            + ([float(values["raw_labels"][i])] if "raw_labels" in values else []))
    print(f"Outer split: {len(train):,} training observations / {len(test):,} test observations", flush=True)
    X, y = values["X"], values["y"]
    searches, oof_by_c = [], {}
    for C in sorted(set(args.c_values)):
        oof = np.full(n, np.nan)
        fold_results = []
        for fold, (fit, val) in enumerate(folds, 1):
            estimator = pipeline(C).fit(X[fit], y[fit])
            if int(estimator[0].n_samples_seen_) != len(fit):
                raise ValueError("Scaler was fitted outside the fold training subset")
            oof[val] = estimator.predict_proba(X[val])[:, 1]
            result = metrics(y[val], oof[val])
            fold_results.append({"fold": fold, "train": stats(values, fit), "validation": stats(values, val), "metrics": result})
            print(f"C={C:g}, fold {fold}/{args.folds}: AP={result['average_precision']:.5f}, ROC AUC={result['roc_auc']:.5f}", flush=True)
        if not np.isfinite(oof[train]).all() or not np.isnan(oof[test]).all():
            raise ValueError("OOF predictions must cover only the outer training set")
        summary = {k: {"mean": float(np.mean([f['metrics'][k] for f in fold_results])),
                       "std": float(np.std([f['metrics'][k] for f in fold_results], ddof=1))}
                   for k in fold_results[0]["metrics"]}
        searches.append({"C": C, "folds": fold_results, "summary": summary})
        oof_by_c[C] = oof
    selected = max(searches, key=lambda r: r["summary"]["average_precision"]["mean"])
    best_C = selected["C"]
    write_predictions(out / "oof_predictions.csv", values, train, oof_by_c[best_C][train], fold_ids)
    print(f"Selected C={best_C:g} from mean cross-validation AP; fitting all outer training observations", flush=True)
    fitted = pipeline(best_C).fit(X[train], y[train])
    test_score = fitted.predict_proba(X[test])[:, 1]
    test_result = metrics(y[test], test_score)
    write_predictions(out / "test_predictions.csv", values, test, test_score)
    bundle = {"pipeline": fitted, "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
              "sklearn_version": sklearn.__version__, "training_source": source,
              "training_scope": "80% of combined split groups; 20% reserved for test",
              "datasets": sorted(np.unique(values["datasets"]).tolist()),
              "data2_label_threshold": args.data2_threshold if args.data2 else None,
              "selected_C": best_C, "seed": args.seed, "cv_folds": args.folds,
              "training_gene_groups": np.unique(values["groups"][train]).tolist(),
              "test_gene_groups": np.unique(values["groups"][test]).tolist()}
    joblib.dump(bundle, out / "baseline_model.joblib")
    overlaps = set(metadata0) & set(metadata1)
    disagreements = sum(metadata0[k]["label"] != metadata1[k]["label"] for k in overlaps)
    report = {"trained_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds"),
              "model": "StandardScaler + L2 logistic regression; unweighted observations; 45 features",
              "all": stats(values, np.arange(n)), "train": stats(values, train), "test": stats(values, test),
              "outer_split": "80/20 of conservative natural gene groups and data2 source-position groups, GroupShuffleSplit" if args.data2 else "80/20 of conservative gene groups, GroupShuffleSplit",
              "data2_label_rule": source.get("data2_label_rule"),
              "seed": args.seed, "cv": {"folds": args.folds, "splitter": "StratifiedGroupKFold",
                                          "seed": args.seed+1, "selection_metric": "mean validation average precision",
                                          "candidates": searches, "selected_C": best_C, "selected_summary": selected["summary"],
                                          "selected_oof_metrics": metrics(y[train], oof_by_c[best_C][train])},
              "test_metrics": test_result,
              "test_metrics_by_dataset": {d: metrics(y[test][values["datasets"][test] == d], test_score[values["datasets"][test] == d])
                                          for d in sorted(np.unique(values["datasets"]).tolist())},
              "constant_test_metrics": metrics(y[test], np.full(len(test), y[train].mean())),
              "overlapping_coordinates": len(overlaps), "overlap_label_disagreements": disagreements,
              "overlap_rule": "retain each dataset observation and its supplied label; same gene group/split/fold",
              "gene_mapping": map_info, "source": source,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
              "verification": {"outer_group_overlap": 0, "outer_natural_transcript_overlap": 0,
                               "outer_condition_id_overlap": len(set(values["transcripts"][train][values["datasets"][train] == "data2"]) & set(values["transcripts"][test][values["datasets"][test] == "data2"])),
                               "data2_position_groups_separated": True if args.data2 else None,
                               "outer_coordinate_overlap": 0, "all_folds_group_separated": True,
                               "scaler_training_rows": int(fitted[0].n_samples_seen_),
                               "test_used_for_parameter_selection": False, "test_included_in_saved_model_fit": False},
              "elapsed_seconds": round(time.perf_counter()-started, 2)}
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Held-out test:", json.dumps(test_result, indent=2), flush=True)
    print(f"Saved combined model and results in {out.resolve()}; {report['elapsed_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
