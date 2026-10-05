"""Shared, conflict-checked transcript-to-gene mapping across datasets."""
import gzip
import re
import pandas as pd


def clean_id(value, strip_version=False):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    return re.sub(r"\.\d+$", "", value) if strip_version else value


def read_gtf_mapping(path):
    rows = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9:
                raise ValueError('GTF must have nine tab-separated columns')
            attrs = dict(re.findall(r'(\w+)\s+"([^"]+)"', fields[8]))
            if 'transcript_id' in attrs and 'gene_id' in attrs:
                rows.append((attrs['transcript_id'], attrs['gene_id']))
    if not rows:
        raise ValueError('No transcript_id/gene_id pairs found in GTF')
    return pd.DataFrame(rows, columns=['transcript_id', 'gene_id']).drop_duplicates()


def load_label_tables(paths, gene_map_path=None, gene_gtf=None, strip_id_version=False):
    required = {'transcript_id', 'transcript_position', 'label'}
    tables, mappings = [], []
    for path in paths:
        table = pd.read_csv(path, dtype={'transcript_id': 'string', 'gene_id': 'string'})
        if not required.issubset(table.columns):
            raise ValueError(f'{path}: Missing label columns: {sorted(required - set(table.columns))}')
        table['transcript_id'] = table.transcript_id.map(clean_id)
        if table.transcript_id.isna().any():
            raise ValueError(f'{path}: Missing transcript IDs')
        positions = pd.to_numeric(table.transcript_position, errors='raise')
        if positions.isna().any() or (positions % 1 != 0).any() or (positions < 0).any():
            raise ValueError(f'{path}: Positions must be nonnegative integers')
        table['transcript_position'] = positions.astype('int64')
        if table.duplicated(['transcript_id', 'transcript_position']).any():
            raise ValueError(f'{path}: Duplicate site coordinates within one dataset')
        tables.append(table)
        if 'gene_id' in table:
            mappings.append(table[['transcript_id', 'gene_id']])
    if gene_map_path:
        mapping = pd.read_csv(gene_map_path, dtype='string')
        if not {'transcript_id', 'gene_id'}.issubset(mapping.columns):
            raise ValueError('--gene-map CSV needs transcript_id and gene_id')
        mappings.append(mapping[['transcript_id', 'gene_id']])
    if gene_gtf:
        mappings.append(read_gtf_mapping(gene_gtf))
    lookup = pd.Series(dtype=object)
    if mappings:
        mapping = pd.concat(mappings, ignore_index=True)
        for column in ('transcript_id', 'gene_id'):
            mapping[column] = mapping[column].map(lambda x: clean_id(x, strip_id_version))
        mapping = mapping.dropna().drop_duplicates()
        conflicts = mapping.groupby('transcript_id').gene_id.nunique()
        if conflicts.gt(1).any():
            raise ValueError('Conflicting transcript-to-gene mapping (including version aliases): '
                             + ', '.join(conflicts[conflicts.gt(1)].index[:10]))
        lookup = mapping.set_index('transcript_id').gene_id
    for table in tables:
        # Canonicalize lookup only; retain original transcript IDs for JSON and predictions.
        table['gene_id'] = table.transcript_id.map(
            lambda x: clean_id(x, strip_id_version)).map(lookup)
    return tables


def write_mapping_audit(tables, out):
    rows, mapped, unmapped = [], [], []
    for source, table in enumerate(tables):
        missing = table.gene_id.isna()
        rows.append({'source_id': source, 'label_sites': len(table),
                     'mapped_sites': int((~missing).sum()), 'unmapped_sites': int(missing.sum()),
                     'mapped_site_fraction': float((~missing).mean()),
                     'transcripts': table.transcript_id.nunique(),
                     'unmapped_transcripts': table.loc[missing, 'transcript_id'].nunique()})
        mapped.append(table.loc[~missing, ['transcript_id', 'gene_id']])
        subset = table.loc[missing, ['transcript_id', 'transcript_position']].copy()
        subset.insert(0, 'source_id', source)
        unmapped.append(subset)
    pd.DataFrame(rows).to_csv(out / 'gene_mapping_coverage.csv', index=False)
    pd.concat(mapped).drop_duplicates().to_csv(out / 'resolved_gene_map.csv', index=False)
    pd.concat(unmapped).to_csv(out / 'unmapped_sites.csv', index=False)
    return rows
