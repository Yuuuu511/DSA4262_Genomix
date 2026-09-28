"""Stream m6Anet JSON lines and summarize read-level signals per site."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = [f"{name}_{offset}" for offset in (-1, 0, 1)
           for name in ("dwell", "sd", "current")]


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def iter_sites(path):
    """Yield (transcript_id, position, seven_mer, reads) from JSONL input."""
    with open_text(path) as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                for transcript, positions in record.items():
                    for position, sequences in positions.items():
                        for seven_mer, reads in sequences.items():
                            yield transcript, int(position), seven_mer, reads
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid JSON format on line {line_no} of {path}: {exc}") from exc


def summarize_site(reads, seven_mer):
    arr = np.asarray(reads, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9 or len(arr) == 0:
        raise ValueError(f"Expected nonempty reads with 9 signals each, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Read signals contain NaN or infinity")
    out = {"read_count": len(arr), "motif": seven_mer[1:6],
           "left_flank": seven_mer[0], "right_flank": seven_mer[-1]}
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    quartiles = np.quantile(arr, [.25, .5, .75], axis=0)
    for index, name in enumerate(SIGNALS):
        out[f"{name}_mean"] = means[index]
        out[f"{name}_std"] = stds[index]
        out[f"{name}_q25"] = quartiles[0, index]
        out[f"{name}_median"] = quartiles[1, index]
        out[f"{name}_q75"] = quartiles[2, index]
    return out


def load_features(path):
    rows = []
    seen = set()
    for transcript, position, seven_mer, reads in iter_sites(path):
        key = (transcript, position)
        if key in seen:
            raise ValueError(f"Duplicate transcript position in input: {key}")
        seen.add(key)
        rows.append({"transcript_id": transcript, "transcript_position": position,
                     **summarize_site(reads, seven_mer)})
    if not rows:
        raise ValueError("No sites found in input JSON")
    return pd.DataFrame.from_records(rows)
