"""Checks for label-free features and the real prediction file contract."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import sklearn

from features import FEATURE_NAMES, FEATURE_VERSION, iter_sites, site_features
from predict import predict
from train import model, read_labels


class BaselineTests(unittest.TestCase):
    def test_feature_values_and_order_invariance(self):
        reads = np.asarray([list(range(1, 10)), list(range(3, 12))], dtype=float)
        x = site_features("AAGACCA", reads)
        np.testing.assert_allclose(x[:9], np.arange(2, 11))
        np.testing.assert_allclose(x[9:18], np.ones(9))
        self.assertAlmostEqual(x[18], np.log(3))
        self.assertEqual(x[19:37].sum(), 1)
        self.assertEqual(x[37:].sum(), 2)
        self.assertEqual(len(x), len(FEATURE_NAMES))
        np.testing.assert_allclose(x, site_features("AAGACCA", reads[::-1]))

    def test_invalid_read_arrays_rejected(self):
        for reads in ([], [[1] * 8], [[float("nan")] * 9], [[-1] * 9]):
            with self.assertRaises(ValueError):
                site_features("AAGACCA", reads)

    def test_duplicate_label_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.csv"
            path.write_text("gene_id,transcript_id,transcript_position,label\ng,t,1,0\ng,t,1,1\n")
            with self.assertRaises(ValueError):
                read_labels(path)

    def test_prediction_contract_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records = [{"t1": {"10": {"AAGACCA": [[1] * 9, [2] * 9]}}},
                       {"t2": {"20": {"AAGACCA": [[3] * 9]}}}]
            data = folder / "data.json"
            data.write_text("".join(json.dumps(r) + "\n" for r in records))
            with gzip.open(folder / "data.json.gz", "wt") as handle:
                handle.write(data.read_text())
            sites = list(iter_sites(data))
            gz_sites = list(iter_sites(folder / "data.json.gz"))
            np.testing.assert_allclose(sites[0][2], gz_sites[0][2])
            fitted = model().fit(np.asarray([s[2] for s in sites]), [0, 1])
            model_path = folder / "model.joblib"
            joblib.dump({"pipeline": fitted, "feature_version": FEATURE_VERSION,
                         "feature_names": FEATURE_NAMES, "sklearn_version": sklearn.__version__}, model_path)
            output = folder / "prediction.csv"
            self.assertEqual(predict(data, model_path, output, batch_size=1), 2)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), ["transcript_id", "transcript_position", "score"])
            self.assertEqual([(r["transcript_id"], r["transcript_position"]) for r in rows],
                             [("t1", "10"), ("t2", "20")])
            self.assertTrue(all(0 <= float(r["score"]) <= 1 for r in rows))
            previous = output.read_text()
            data.write_text(data.read_text() + json.dumps(records[0]) + "\n")
            with self.assertRaises(ValueError):
                predict(data, model_path, output, batch_size=1)
            self.assertEqual(previous, output.read_text())
            self.assertEqual(list(folder.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
