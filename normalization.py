"""Training-negative motif reference and explicit read-distribution features."""
import numpy as np
from features import SIGNALS, summarize_site


class Reservoir:
    def __init__(self, capacity, rng):
        self.values = np.empty((capacity, 9))
        self.capacity, self.rng, self.count = capacity, rng, 0

    def add(self, rows):
        for row in rows:
            if self.count < self.capacity:
                self.values[self.count] = row
            else:
                index = int(self.rng.integers(self.count + 1))
                if index < self.capacity:
                    self.values[index] = row
            self.count += 1

    def array(self):
        return self.values[:min(self.count, self.capacity)]


def location_scale(rows):
    center = np.median(rows, axis=0)
    scale = 1.4826 * np.median(np.abs(rows - center), axis=0)
    # Avoid division by zero for near-constant features.
    scale = np.maximum(scale, np.maximum(np.std(rows, axis=0) * 0.01, 1e-6))
    return center, scale


def finish_reference(reservoirs, global_reservoir, counts, min_sites):
    if not global_reservoir.count:
        raise ValueError("No training-negative reads available for motif reference")
    center, scale = location_scale(global_reservoir.array())
    refs = {}
    for motif, reservoir in reservoirs.items():
        if counts[motif] >= min_sites:
            c, s = location_scale(reservoir.array())
            refs[motif] = {"center": c, "scale": s, "sites": counts[motif],
                           "sampled_reads": len(reservoir.array())}
    return {"motifs": refs, "global_center": center, "global_scale": scale,
            "negative_site_counts": dict(counts), "minimum_reference_sites": min_sites,
            "reference_definition": "training label=0 only; up to 20 reads/site; median/MAD"}


def normalized_site(reads, seven_mer, reference):
    arr = np.asarray(reads, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 9 or not len(arr) or not np.isfinite(arr).all():
        raise ValueError("Expected nonempty finite read matrix with 9 features")
    motif = seven_mer[1:6]
    ref = reference["motifs"].get(motif)
    center = ref["center"] if ref else reference["global_center"]
    scale = ref["scale"] if ref else reference["global_scale"]
    z = (arr - center) / scale
    summaries = summarize_site(z, seven_mer)
    out = {"norm_" + k: v for k, v in summaries.items()
           if k not in ("read_count", "motif", "left_flank", "right_flank")}
    out["reference_fallback"] = int(ref is None)
    for j, name in enumerate(SIGNALS):
        out[f"dist_{name}_upper2"] = float(np.mean(z[:, j] > 2))
        out[f"dist_{name}_lower2"] = float(np.mean(z[:, j] < -2))
        out[f"dist_{name}_abs2"] = float(np.mean(np.abs(z[:, j]) > 2))
    # Joint per-read deviation captures relationships that marginal quantiles omit.
    joint = np.sqrt(np.mean(z * z, axis=1))
    out["dist_joint_median"] = float(np.median(joint))
    out["dist_joint_q90"] = float(np.quantile(joint, .9))
    out["dist_joint_above2"] = float(np.mean(joint > 2))
    out["dist_multi_feature_tail"] = float(np.mean((np.abs(z) > 2).sum(axis=1) >= 3))
    return out
