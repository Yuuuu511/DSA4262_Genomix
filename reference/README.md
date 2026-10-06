# Gene-group snapshot

`transcript_gene_map.csv` preserves the original mapping: 6,352 transcripts, 4,486 preferred gene IDs and 4,379 conservative split groups. Rows contain transcript, preferred gene, group and mapping source.

Groups conservatively join gene IDs linked by a transcript mapping or shared Ensembl 96 gene name, keeping alternate patch mappings together. A group is not necessarily one current biological gene. Modification label values do not create groups.

`provenance.json` records mapping/source checksums, URLs, counts and label-file hashes. Default training verifies these fingerprints, exact transcript coverage, data0 gene agreement and consistent gene-group assignments. Different natural-data metadata requires rebuilding the map.

`ensembl_lookup.json` is the saved REST response for 54 additional transcripts. Reuse this snapshot for historical reproduction; live annotations can change.

To rebuild from raw references, obtain the SG-NEx Ensembl 91 table and Ensembl 96 GTF with patch contigs from the URLs in `provenance.json`, verify their hashes, and run:

```bash
python train_combined.py \
  --data0 data/data0/dataset0.json.gz \
  --labels0 data/data0/data.info.labelled \
  --data1 data/data1/dataset1.json.gz \
  --labels1 data/data1/data.info \
  --annotation PATH_TO_ENSEMBL91_ANNOTATION.txt \
  --lookup reference/ensembl_lookup.json \
  --patch-gtf PATH_TO_ENSEMBL96_PATCH_GTF.gtf.gz
```

Supply all three raw-reference arguments together. This writes the map to `artifacts/combined/transcript_gene_map.csv`; the default compact route needs no downloads. Add data2 arguments to include its source-position groups.

The SG-NEx table headers do not match its first data columns; the parser uses ENST/ENSG prefixes. Retired transcript `ENST00000627066` requires the patch annotation. See [the mapping derivation](../docs/combined_training.md).
