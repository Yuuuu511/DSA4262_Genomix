from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. Local file paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "merged"

DATA_PATHS = {
    "data0": {
        "json": BASE_DIR / "data0" / "dataset0.json.gz",
        "labels": BASE_DIR / "data0" / "data.info.labelled",
    },
    "data1": {
        "json": BASE_DIR / "data1" / "dataset1.json.gz",
        "labels": BASE_DIR / "data1" / "data.info",
    },
    "data2": {
        "json": BASE_DIR / "data2" / "dataset2.json.gz",
        "labels": BASE_DIR / "data2" / "data.info",
    },
}

KEYS = ["transcript_id", "transcript_position"]

FEATURE_COLS = (
    ["read_count"]
    + [f"mean_{i}" for i in range(1, 10)]
    + [f"std_{i}" for i in range(1, 10)]
    + [f"median_{i}" for i in range(1, 10)]
)


# ============================================================
# 2. Load labels
# ============================================================

def load_labels(path, dataset_name):
    labels = pd.read_csv(path)
    labels.columns = labels.columns.str.strip()

    required = set(KEYS + ["label"])
    missing = required - set(labels.columns)

    if missing:
        raise ValueError(f"{dataset_name} 缺少列：{missing}")

    positions = pd.to_numeric(
        labels["transcript_position"], errors="raise"
    )
    if positions.isna().any() or (positions % 1 != 0).any():
        raise ValueError(f"{dataset_name} 存在无效的位点坐标")

    labels["transcript_position"] = positions.astype("int64")
    labels["label_original"] = pd.to_numeric(
        labels["label"], errors="raise"
    )

    if not labels["label_original"].between(0, 1).all():
        raise ValueError(f"{dataset_name} 存在缺失或不在 [0, 1] 内的标签")

    if labels[KEYS].isna().any().any():
        raise ValueError(f"{dataset_name} 存在缺失的位点标识")

    if labels.duplicated(KEYS).any():
        raise ValueError(f"{dataset_name} 标签文件存在重复位点")

    if dataset_name == "data2":
        # Assign values equal to or greater than 0.5 to class 1.
        labels["label"] = (
            labels["label_original"] >= 0.5
        ).astype(int)
    else:
        if not labels["label_original"].isin([0, 1]).all():
            raise ValueError(f"{dataset_name} 存在非二分类标签")
        labels["label"] = labels["label_original"].astype(int)

    # start/end are offsets in the original JSON file.
    # They are not needed in the merged feature table.
    keep = KEYS + ["label_original", "label"]
    if "gene_id" in labels.columns:
        keep.append("gene_id")

    return labels[keep].copy()


# ============================================================
# 3. Aggregate reads at each site into 28 features
# ============================================================

def load_features(path, dataset_name):
    rows = []

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            for transcript_id, positions in record.items():
                for position, sequences in positions.items():
                    if len(sequences) != 1:
                        raise ValueError(
                            f"{dataset_name}: {transcript_id}:{position} "
                            "包含多个或零个序列，需要单独检查"
                        )

                    sequence, reads = next(iter(sequences.items()))
                    reads = np.asarray(reads, dtype=float)

                    if (
                        reads.ndim != 2
                        or reads.shape[0] == 0
                        or reads.shape[1] != 9
                    ):
                        raise ValueError(
                            f"{dataset_name}: {transcript_id}:{position} "
                            f"reads 形状异常：{reads.shape}"
                        )

                    if not np.isfinite(reads).all():
                        raise ValueError(
                            f"{dataset_name}: {transcript_id}:{position} "
                            "存在缺失值或无穷值"
                        )

                    row = {
                        "transcript_id": transcript_id,
                        "transcript_position": int(position),
                        "sequence": sequence,
                        "read_count": reads.shape[0],
                    }

                    summaries = {
                        "mean": np.mean(reads, axis=0),
                        "std": np.std(reads, axis=0, ddof=0),
                        "median": np.median(reads, axis=0),
                    }

                    for name, values in summaries.items():
                        for i, value in enumerate(values, start=1):
                            row[f"{name}_{i}"] = value

                    rows.append(row)

            if line_number % 20000 == 0:
                print(
                    f"  {dataset_name}: 已读取 {line_number:,} 行",
                    flush=True,
                )

    features = pd.DataFrame(rows)

    if features.empty:
        raise ValueError(f"{dataset_name} 没有读取到特征")

    if features.duplicated(KEYS).any():
        raise ValueError(f"{dataset_name} JSON 中存在重复位点")

    return features


# ============================================================
# 4. Match labels and combine the three datasets
# ============================================================

def main():
    for paths in DATA_PATHS.values():
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"找不到文件：{path}")

    labels_by_dataset = {
        name: load_labels(paths["labels"], name)
        for name, paths in DATA_PATHS.items()
    }

    # Build a transcript_id -> gene_id mapping from data0.
    gene_pairs = (
        labels_by_dataset["data0"][["transcript_id", "gene_id"]]
        .dropna()
        .drop_duplicates()
    )

    if gene_pairs["transcript_id"].duplicated().any():
        raise ValueError("data0 中同一个 transcript_id 对应多个 gene_id")

    gene_map = gene_pairs.set_index("transcript_id")["gene_id"]

    merged_parts = []

    for name, paths in DATA_PATHS.items():
        print(f"\n正在处理 {name} ...", flush=True)

        features = load_features(paths["json"], name)
        labels = labels_by_dataset[name]

        # Match features and labels within each dataset first.
        # This prevents cross-dataset matches for shared sites.
        part = features.merge(
            labels,
            on=KEYS,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        unmatched = part["_merge"] != "both"
        if unmatched.any():
            counts = part.loc[unmatched, "_merge"].value_counts()
            raise ValueError(
                f"{name} 特征与标签未完全匹配：\n{counts}"
            )

        part = part.drop(columns="_merge")
        part.insert(0, "source_dataset", name)

        if "gene_id" not in part.columns:
            part["gene_id"] = part["transcript_id"].map(gene_map)
        else:
            part["gene_id"] = part["gene_id"].fillna(
                part["transcript_id"].map(gene_map)
            )

        # Include the dataset name to create a unique sample identifier.
        # Shared sites from different datasets remain separate rows.
        part.insert(
            1,
            "sample_id",
            part["source_dataset"]
            + ":"
            + part["transcript_id"]
            + ":"
            + part["transcript_position"].astype(str),
        )

        print(f"  匹配成功：{len(part):,} 个位点")
        print(f"  label = 0：{(part['label'] == 0).sum():,}")
        print(f"  label = 1：{(part['label'] == 1).sum():,}")
        print(f"  gene_id 缺失：{part['gene_id'].isna().sum():,}")

        merged_parts.append(part)

    merged = pd.concat(merged_parts, ignore_index=True)

    column_order = [
        "source_dataset",
        "sample_id",
        "gene_id",
        "transcript_id",
        "transcript_position",
        "sequence",
        "label_original",
        "label",
    ] + FEATURE_COLS

    merged = merged[column_order]

    if merged["sample_id"].duplicated().any():
        raise ValueError("合并后 sample_id 不唯一")

    expected_rows = sum(len(df) for df in labels_by_dataset.values())
    if len(merged) != expected_rows:
        raise ValueError("合并后的行数不符合预期")

    summary = (
        merged.groupby(["source_dataset", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
        .rename(columns={0: "label_0", 1: "label_1"})
    )
    summary["total"] = summary["label_0"] + summary["label_1"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "merged_site_features.csv"
    summary_path = OUTPUT_DIR / "label_summary.csv"

    merged.to_csv(output_path, index=False)
    summary.to_csv(summary_path)

    print("\n合并完成：")
    print(summary.to_string())
    print(f"\n总位点数：{len(merged):,}")
    print(f"特征数量：{len(FEATURE_COLS)}")
    print(f"\n数据文件：{output_path}")
    print(f"统计文件：{summary_path}")


if __name__ == "__main__":
    main()