"""Prepare ALL data2 sites for official m6Anet; no labels or manifest needed.

Add synthetic per-site read IDs to nine-column course signals and rebuild the
byte offsets in data.info. Exclude sites with fewer than 20 reads, recording them.
"""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import tempfile


def prepare(data, output_dir):
    data = Path(data).resolve()
    out = Path(output_dir).resolve()
    if data == out or out in data.parents:
        raise ValueError('Output directory must not contain the input file')
    if out.exists() and any(p.name != '.DS_Store' for p in out.iterdir()):
        raise FileExistsError('Data2 m6Anet input directory is nonempty')
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    skipped = []
    prepared = 0
    opener = gzip.open if str(data).endswith('.gz') else open
    with tempfile.TemporaryDirectory(prefix='data2_prepare_', dir=out.parent) as temp:
        temp = Path(temp)
        with opener(data, 'rt', encoding='utf-8') as handle, \
                (temp / 'data.json').open('wb') as signals, \
                (temp / 'data.info').open('w', newline='', encoding='utf-8') as info:
            writer = csv.writer(info)
            writer.writerow(['transcript_id', 'transcript_position', 'n_reads', 'start', 'end'])
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                for tx, positions in json.loads(line).items():
                    if not isinstance(tx, str) or not tx.strip():
                        raise ValueError(f'Invalid transcript ID at line {line_no}')
                    for position, sequences in positions.items():
                        pos = int(position)
                        key = (tx, pos)
                        if key in seen or len(sequences) != 1:
                            raise ValueError(f'Duplicate site or unexpected sequences: {key}')
                        seen.add(key)
                        seq, reads = next(iter(sequences.items()))
                        seq = seq.upper().replace('U', 'T')
                        if len(seq) != 7 or set(seq) - set('ACGT'):
                            raise ValueError(f'Invalid seven-base sequence: {key}')
                        if not isinstance(reads, list) or not reads:
                            raise ValueError(f'Empty or malformed reads: {key}')
                        for read in reads:
                            if not isinstance(read, list) or len(read) != 9 or not all(
                                    isinstance(x, (int, float)) and math.isfinite(x) for x in read):
                                raise ValueError(f'Reads must contain nine finite numbers: {key}')
                        if len(reads) < 20:
                            skipped.append({'transcript_id': tx, 'transcript_position': pos,
                                            'n_reads': len(reads)})
                            continue
                        indexed = [read + [index] for index, read in enumerate(reads)]
                        encoded = (json.dumps({tx: {str(pos): {seq: indexed}}},
                                              separators=(',', ':'), ensure_ascii=True) + '\n').encode('ascii')
                        start = signals.tell()
                        signals.write(encoded)
                        writer.writerow([tx, pos, len(reads), start, signals.tell()])
                        prepared += 1
        if not prepared:
            raise ValueError('No sites meet the m6Anet minimum of 20 reads')
        report = {'dataset': 'data2', 'requested_sites': len(seen), 'prepared_sites': prepared,
                  'minimum_reads': 20, 'excluded_low_coverage_sites': len(skipped),
                  'skipped_sites': skipped,
                  'notes': ['All raw data2 sites considered; no split-manifest filtering.',
                            'Original identifiers and integer positions preserved, including position zero.',
                            'Synthetic per-site read IDs appended only for m6Anet input compatibility.',
                            'No labels used, and no model fitted.']}
        (temp / 'preparation_report.json').write_text(json.dumps(report, indent=2))
        out.mkdir(parents=True, exist_ok=True)
        for name in ['data.json', 'data.info', 'preparation_report.json']:
            os.replace(temp / name, out / name)
    print(f'Data2 requested sites: {len(seen):,}')
    print(f'Prepared sites: {prepared:,}')
    print(f'Excluded sites with fewer than 20 reads: {len(skipped):,}')
    print(f'Input directory: {out}')
    return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root.parent / 'data2/dataset2.json.gz')
    parser.add_argument('--output-dir', type=Path, default=root / 'evaluation/m6anet_inputs/data2')
    args = parser.parse_args()
    prepare(args.data, args.output_dir)


if __name__ == '__main__':
    main()
