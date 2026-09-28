"""Generate artificial signals solely to smoke-test the scripts and CSV format."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(__file__).parent / "example"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
rows = []
with (out / "example.json").open("w") as f:
    for g in range(80):
        transcript = f"ENST_EXAMPLE_{g:04d}"
        for j in range(5):
            position = 100 + j * 20
            label = int(rng.random() < .12)
            signal = rng.normal(size=(rng.integers(12, 30), 9))
            signal[:, 5] += 1.7 * label
            signal[:, 2::3] += 110
            signal[:, 0::3] = np.exp(signal[:, 0::3] - 4)
            signal[:, 1::3] = np.exp(signal[:, 1::3] + 1)
            f.write(json.dumps({transcript: {str(position): {"AAGACCA": signal.tolist()}}}) + "\n")
            rows.append((f"ENSG_EXAMPLE_{g:04d}", transcript, position, label))
pd.DataFrame(rows, columns=["gene_id", "transcript_id", "transcript_position", "label"]).to_csv(
    out / "example.info", index=False)
print(f"Generated {len(rows)} synthetic sites in {out}")
