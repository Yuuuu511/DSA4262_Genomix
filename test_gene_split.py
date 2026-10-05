"""Regression checks for cross-dataset leakage and mapping integrity."""
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from label_utils import load_label_tables
from train_experiments import split_data, write_split_audit


class GeneSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def table(self, name, rows):
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_shared_mapping_and_different_labels(self):
        a = self.table('a.csv', [{'transcript_id': 'T1', 'transcript_position': 1,
                                 'gene_id': 'G1', 'label': 0}])
        b = self.table('b.csv', [{'transcript_id': 'T1', 'transcript_position': 1, 'label': 1}])
        tables = load_label_tables([a, b])
        self.assertEqual(tables[1].gene_id.tolist(), ['G1'])
        self.assertEqual([t.label.iloc[0] for t in tables], [0, 1])

    def test_conflict_rejected(self):
        a = self.table('a.csv', [{'transcript_id': 'T1', 'transcript_position': 1,
                                 'gene_id': 'G1', 'label': 0}])
        mapping = self.table('map.csv', [{'transcript_id': 'T1', 'gene_id': 'G2'}])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            load_label_tables([a], mapping)

    def test_gtf_versions_preserve_site_ids(self):
        a = self.table('a.csv', [{'transcript_id': 'T1.2', 'transcript_position': 1, 'label': 0},
                               {'transcript_id': 'UNKNOWN', 'transcript_position': 2, 'label': 1}])
        gtf = self.root / 'ref.gtf'
        gtf.write_text('chr1\tx\texon\t1\t20\t.\t+\t.\tgene_id "G1.4"; transcript_id "T1";\n')
        exact = load_label_tables([a], gene_gtf=str(gtf))[0]
        self.assertTrue(exact.gene_id.isna().all())
        normalized = load_label_tables([a], gene_gtf=str(gtf), strip_id_version=True)[0]
        self.assertEqual(normalized.gene_id.iloc[0], 'G1')
        self.assertEqual(normalized.transcript_id.iloc[0], 'T1.2')
        self.assertTrue(pd.isna(normalized.gene_id.iloc[1]))

    def test_joint_split_and_manifest_reuse(self):
        data = pd.DataFrame([{'source_id': source, 'gene_id': f'G{g}',
                              'transcript_id': f'T{g}_{iso}', 'transcript_position': 1,
                              'label': source}
                             for source in (0, 1) for g in range(30) for iso in (0, 1)])
        assigned, manifest = split_data(data.copy(), 'gene_id', self.root, {'test': 1})
        write_split_audit(assigned, 'gene_id', self.root)
        self.assertTrue(assigned.groupby('gene_id').split.nunique().eq(1).all())
        self.assertEqual(len(assigned), len(data))
        checks = json.loads((self.root / 'split_checks.json').read_text())
        self.assertEqual(checks['cross_dataset_discordant_coordinates'], 60)
        reused, _ = split_data(data.copy(), 'gene_id', self.root, {'test': 1})
        self.assertEqual(assigned.split.tolist(), reused.split.tolist())
        with self.assertRaisesRegex(ValueError, 'Inputs or split mode changed'):
            split_data(data.copy(), 'gene_id', self.root, {'test': 2})

    def test_within_dataset_duplicates_rejected(self):
        row = {'transcript_id': 'T1', 'transcript_position': 1, 'label': 0}
        a = self.table('a.csv', [row, row])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_label_tables([a])


if __name__ == '__main__':
    unittest.main()
