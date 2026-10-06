"""Ensure repeated/related observations stay together through outer and CV splits."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_combined import GeneGroups, extract_data2, make_splits, threshold_labels


class CombinedTests(unittest.TestCase):
    def test_data2_extraction_preserves_fractions_and_groups_repeated_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            data, labels = Path(folder) / "data.json.gz", Path(folder) / "labels.csv"
            records = [{"tx_id_1": {"0": {"AAGACCA": [[1] * 9]}}},
                       {"tx_id_5": {"0": {"AAGACCA": [[2] * 9]}}}]
            with gzip.open(data, "wt") as handle:
                handle.write("".join(json.dumps(r) + "\n" for r in records))
            with labels.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["transcript_id", "transcript_position", "label", "n_reads"])
                writer.writerows([["tx_id_1", 0, 0.95, 1], ["tx_id_5", 0, 0.25, 1]])
            values = extract_data2(data, labels, 0.5)
            np.testing.assert_array_equal(values["y"], [1, 0])
            np.testing.assert_allclose(values["raw_labels"], [0.95, 0.25])
            self.assertEqual(values["groups"][0], values["groups"][1])
            records[1]["tx_id_5"]["0"] = {"AAGACCG": [[2] * 9]}
            with gzip.open(data, "wt") as handle:
                handle.write("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaises(ValueError):
                extract_data2(data, labels, 0.5)

    def test_threshold_includes_exactly_half_and_rejects_invalid_fractions(self):
        np.testing.assert_array_equal(threshold_labels([0, 0.25, 0.499999, 0.5, 0.7, 0.75, 0.95, 1]),
                                      [0, 0, 0, 1, 1, 1, 1, 1])
        for labels in ([float("nan")], [float("inf")], [-0.01], [1.01]):
            with self.assertRaises(ValueError):
                threshold_labels(labels)

    def test_gene_aliases_merge_transitively(self):
        groups = GeneGroups()
        groups.union("gene_B", "gene_C")
        groups.union("gene_A", "gene_B")
        self.assertEqual(groups.find("gene_A"), groups.find("gene_C"))
        self.assertNotEqual(groups.find("unrelated"), groups.find("gene_A"))

    def test_shared_sites_and_aliases_stay_in_same_split_and_fold(self):
        # Two dataset observations per site, with disagreeing labels on some sites.
        groups, genes, transcripts, positions, labels = [], [], [], [], []
        for gene in range(50):
            for site in range(8):
                for dataset in range(2):
                    groups.append(f"group_{gene:02d}")
                    genes.append(f"gene_{gene:02d}_alias_{dataset}")
                    transcripts.append(f"transcript_{gene:02d}")
                    positions.append(str(site))
                    labels.append(int((site + dataset) % 4 == 0))
        values = {"X": np.zeros((len(labels), 45)), "y": np.asarray(labels),
                  "groups": np.asarray(groups), "genes": np.asarray(genes),
                  "transcripts": np.asarray(transcripts), "positions": np.asarray(positions)}
        train, test, folds = make_splits(values)
        self.assertEqual(len(set(values["groups"][train])), 40)
        self.assertEqual(len(set(values["groups"][test])), 10)
        assignment = {i: "test" for i in test}
        for fold_id, (fit, val) in enumerate(folds, 1):
            self.assertFalse(set(values["groups"][fit]) & set(values["groups"][val]))
            self.assertFalse(set(fit) & set(test))
            self.assertFalse(set(val) & set(test))
            for i in val:
                self.assertNotIn(i, assignment)
                assignment[i] = fold_id
        self.assertEqual(len(assignment), len(labels))
        for i in range(0, len(labels), 2):
            self.assertEqual(assignment[i], assignment[i+1])
        again_train, again_test, _ = make_splits(values)
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(test, again_test)

    def test_data2_condition_ids_can_repeat_but_source_positions_cannot_cross(self):
        # One identifier per label condition, repeated at each source position.
        labels = np.tile([1, 1, 1, 1, 1, 0, 0], 50)
        values = {"X": np.zeros((350, 45)), "y": labels,
                  "groups": np.repeat([f"data2_position_{p}" for p in range(50)], 7),
                  "genes": np.full(350, ""), "datasets": np.full(350, "data2"),
                  "transcripts": np.tile([f"tx_id_{c}" for c in range(7)], 50),
                  "positions": np.repeat([str(p) for p in range(50)], 7)}
        train, test, folds = make_splits(values)
        self.assertEqual(set(values["transcripts"][train]), set(values["transcripts"][test]))
        self.assertFalse(set(values["positions"][train]) & set(values["positions"][test]))
        for fit, val in folds:
            self.assertFalse(set(values["positions"][fit]) & set(values["positions"][val]))
            self.assertFalse(set(val) & set(test))


if __name__ == "__main__":
    unittest.main()
