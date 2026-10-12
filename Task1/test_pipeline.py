"""Small reproducibility and input/output contract tests; no real retraining."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from predict import predict_file, load_model
from train import read_manifest
from validate_submission import validate
from xgb_features import site_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts/selected_xgb.joblib'


class PipelineTests(unittest.TestCase):
    def test_model_manifest_matches_package(self):
        bundle = load_model(MODEL)
        self.assertTrue(bundle['manifest'].equals(read_manifest(ROOT / 'splits/split_manifest.csv')))
        self.assertEqual(bundle['training_data'], 'data0 train+validation; RF original manifest')

    def test_real_model_prediction_and_batch_invariance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for size in [1, 7, 2048]:
                output = root / f'p{size}.csv'
                count = predict_file(ROOT / 'example/test.json.gz', MODEL, output, size)
                self.assertEqual(count, 20)
                self.assertEqual(validate(ROOT / 'example/test.json.gz', output)['status'], 'PASS')
                actual = pd.read_csv(output, dtype={'transcript_id': str})
                expected = pd.read_csv(ROOT / 'example/expected_predictions.csv', dtype={'transcript_id': str})
                pd.testing.assert_frame_equal(actual.iloc[:, :2], expected.iloc[:, :2])
                np.testing.assert_allclose(actual.score, expected.score, atol=1e-6, rtol=0)

    def test_plain_json_and_position_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'sample.json'
            reads = [[.01, 2, 100, .02, 3, 110, .03, 4, 90]] * 3
            record = {'tx_id_0': {'0': {'AAGACCA': reads}}}
            data.write_text(json.dumps(record) + '\n')
            output = root / 'p.csv'
            predict_file(data, MODEL, output)
            self.assertEqual(validate(data, output)['rows'], 1)
            frame = pd.read_csv(output)
            self.assertEqual(frame.transcript_position.iloc[0], 0)
            with self.assertRaises(FileExistsError):
                predict_file(data, MODEL, output)
            data.write_text(json.dumps(record) + '\n' + json.dumps(record) + '\n')
            with self.assertRaises(ValueError):
                predict_file(data, MODEL, root / 'duplicate.csv')
            self.assertFalse((root / 'duplicate.csv').exists())

    def test_invalid_reads(self):
        for reads in [[], [[1] * 8], [[float('nan')] * 9]]:
            with self.assertRaises(ValueError):
                site_features('AAGACCA', reads)


if __name__ == '__main__':
    unittest.main()
