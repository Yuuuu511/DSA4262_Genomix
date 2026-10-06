"""Shared, label-free site features for training and prediction."""
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

FEATURE_VERSION = 1
MOTIFS = [d + r + "AC" + h for d, r, h in itertools.product("AGT", "AG", "ACT")]
SIGNALS = [f"{position}_{signal}" for position in ("minus1", "central", "plus1")
           for signal in ("dwell", "current_sd", "current_mean")]
FEATURE_NAMES = ([f"mean_{s}" for s in SIGNALS] + [f"sd_{s}" for s in SIGNALS]
                 + ["log1p_read_count"] + [f"motif_{m}" for m in MOTIFS]
                 + [f"{flank}_{base}" for flank in ("left_base", "right_base") for base in "ACGT"])


def site_features(sequence, reads):
    sequence = sequence.upper().replace("U", "T")
    if len(sequence) != 7 or any(b not in "ACGT" for b in sequence):
        raise ValueError(f"Expected an unambiguous 7-mer, got {sequence!r}")
    if sequence[1:6] not in MOTIFS:
        raise ValueError(f"Central 5-mer is not DRACH: {sequence!r}")
    values = np.asarray(reads, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 9:
        raise ValueError(f"Expected one or more reads with 9 features, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Read features contain NaN or infinity")
    if (values < 0).any():
        raise ValueError("Signal features must be non-negative")
    context = ([float(sequence[1:6] == m) for m in MOTIFS]
               + [float(sequence[0] == b) for b in "ACGT"]
               + [float(sequence[6] == b) for b in "ACGT"])
    return np.concatenate((values.mean(axis=0), values.std(axis=0, ddof=0),
                           [np.log1p(len(values))], context))


def iter_sites(path):
    """Stream JSON-lines input without loading all nanopore reads into memory."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or not record:
                    raise ValueError("Expected a non-empty transcript mapping")
                for transcript, positions in record.items():
                    if not positions:
                        raise ValueError("Empty position mapping")
                    for position, contexts in positions.items():
                        if not position.isdigit() or str(int(position)) != position:
                            raise ValueError(f"Non-canonical transcript position {position!r}")
                        if len(contexts) != 1:
                            raise ValueError("Expected exactly one sequence context per site")
                        sequence, reads = next(iter(contexts.items()))
                        yield transcript, position, site_features(sequence, reads)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"{path}, line {line_number}: {exc}") from exc
