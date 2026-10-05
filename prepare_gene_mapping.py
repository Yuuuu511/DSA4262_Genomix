"""Fast mapping preflight: labels only, no JSON scan and no model training."""
import argparse
from pathlib import Path
import pandas as pd
from label_utils import load_label_tables, write_mapping_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--gene-map')
    parser.add_argument('--gene-gtf')
    parser.add_argument('--strip-id-version', action='store_true')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = load_label_tables(args.labels, args.gene_map, args.gene_gtf, args.strip_id_version)
    rows = write_mapping_audit(tables, out)
    print(pd.DataFrame(rows).to_string(index=False))
    if any(row['unmapped_sites'] for row in rows):
        raise SystemExit('Unmapped sites remain. Inspect unmapped_sites.csv and complete the mapping.')


if __name__ == '__main__':
    main()
