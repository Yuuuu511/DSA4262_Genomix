"""Validate the distributable models, reference snapshot and portable CLI."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from train import digest
from train_combined import load_gene_map

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_published_models_and_reference_match_checksums(self):
        manifest = json.loads((ROOT / "models/manifest.json").read_text())
        self.assertEqual(manifest["default_model"], "models/baseline_model.joblib")
        for record in manifest["models"]:
            path = ROOT / record["file"]
            self.assertEqual(digest(path), record["sha256"])
            bundle = joblib.load(path)
            self.assertEqual(bundle["sklearn_version"], record["scikit_learn_version"])
            self.assertEqual(bundle["pipeline"][1].classes_.tolist(), [0, 1])
        provenance = json.loads((ROOT / "reference/provenance.json").read_text())
        self.assertEqual(digest(ROOT / "reference/transcript_gene_map.csv"), provenance["mapping_sha256"])
        self.assertEqual(digest(ROOT / "reference/ensembl_lookup.json"), provenance["reference_sources"]["lookup"]["sha256"])
        latest = joblib.load(ROOT / manifest["default_model"])
        self.assertEqual(latest["datasets"], ["data0", "data1", "data2"])
        self.assertEqual(latest["data2_label_threshold"], 0.5)
        self.assertEqual(latest["pipeline"][0].n_samples_seen_, 169525)
        self.assertFalse(set(latest["training_gene_groups"]) & set(latest["test_gene_groups"]))

    def test_default_cli_works_outside_repository_and_matches_example(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "prediction.csv"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "predict.py"), "--data", str(ROOT / "example/test_data.json.gz"),
                 "--output", str(output), "--batch-size", "3"],
                cwd=folder, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with output.open() as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["transcript_id", "transcript_position", "score"])
                actual = list(reader)
            with (ROOT / "example/expected_predictions.csv").open() as handle:
                expected = list(csv.DictReader(handle))
            self.assertEqual(len(actual), 20)
            self.assertEqual([(r["transcript_id"], r["transcript_position"]) for r in actual],
                             [(r["transcript_id"], r["transcript_position"]) for r in expected])
            scores = np.asarray([float(r["score"]) for r in actual])
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(((scores >= 0) & (scores <= 1)).all())
            np.testing.assert_allclose(scores, [float(r["score"]) for r in expected], rtol=1e-10, atol=1e-12)

    def test_mapping_snapshot_rejects_changed_labels_or_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            labels0, labels1 = folder / "labels0.csv", folder / "labels1.csv"
            labels0.write_text("gene_id,transcript_id,transcript_position,label\nENSG1,ENST1,1,0\n")
            labels1.write_text("transcript_id,transcript_position,label\nENST2,2,1\n")
            mapping, provenance = folder / "map.csv", folder / "provenance.json"
            mapping.write_text("transcript_id,gene_id,gene_group,mapping_source\n"
                               "ENST1,ENSG1,ENSG1,data0_labels\nENST2,ENSG2,ENSG1,annotation\n")
            provenance.write_text(json.dumps({"mapping_sha256": digest(mapping),
                "label_sources": {"labels0": {"sha256": digest(labels0)}, "labels1": {"sha256": digest(labels1)}},
                "gene_mapping": {"transcripts": 2, "preferred_gene_ids": 2, "conservative_gene_groups": 1,
                                 "preferred_gene_source_counts": {"data0_labels": 1, "annotation": 1}}}))
            reference, metadata = {("ENST1", "1"): ("ENSG1", 0)}, {("ENST2", "2"): {"label": "1"}}
            arguments = (reference, metadata, mapping, provenance, labels0, labels1)
            genes, groups, _, _ = load_gene_map(*arguments)
            self.assertEqual(genes["ENST2"], "ENSG2")
            self.assertEqual(groups["ENST1"], groups["ENST2"])
            original = labels1.read_text()
            labels1.write_text(original.replace(",1\n", ",0\n"))
            with self.assertRaisesRegex(ValueError, "labels1 differs"):
                load_gene_map(*arguments)
            labels1.write_text(original)
            mapping.write_text(mapping.read_text().replace("ENSG2", "ENSG3"))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_gene_map(*arguments)


if __name__ == "__main__":
    unittest.main()
