
from pathlib import Path
import pandas as pd

root = Path("sg_nex_prediction_results/MCF7")
files = sorted(root.glob("*.csv"))

assert len(files) == 2, f"Expected 2 files, found {len(files)}"

a = pd.read_csv(files[0])
b = pd.read_csv(files[1])

print("FILES")
for path, df in zip(files, [a, b]):
    print(f"\n{path.name}")
    print("Sites:", len(df))
    print("Unique transcripts:", df.transcript_id.nunique())
    print("Position range:",
          df.transcript_position.min(),
          df.transcript_position.max())
    print("First 5 rows:")
    print(df.head().to_string(index=False))
    print("Non-ENST transcript examples:",
          df.loc[
              ~df.transcript_id.astype(str).str.startswith("ENST"),
              "transcript_id"
          ].head(10).tolist())

ids_a = set(a.transcript_id)
ids_b = set(b.transcript_id)

print("\nTRANSCRIPT ID OVERLAP")
print("Run A transcripts:", len(ids_a))
print("Run B transcripts:", len(ids_b))
print("Shared transcripts:", len(ids_a & ids_b))
print("Only A examples:", sorted(ids_a - ids_b)[:10])
print("Only B examples:", sorted(ids_b - ids_a)[:10])

keys = ["transcript_id", "transcript_position"]

shared = a[keys].merge(b[keys], on=keys)
print("\nExact shared sites:", len(shared))

# Check a possible systematic position shift
common = sorted(ids_a & ids_b)
if common:
    print("\nPosition overlap for example transcripts:")
    for transcript in common[:10]:
        p1 = set(a.loc[a.transcript_id == transcript,
                       "transcript_position"])
        p2 = set(b.loc[b.transcript_id == transcript,
                       "transcript_position"])
        print(
            transcript,
            "A:", len(p1),
            "B:", len(p2),
            "Shared:", len(p1 & p2),
            "A examples:", sorted(p1)[:5],
            "B examples:", sorted(p2)[:5]
        )
