"""Focused QC regression and failure-path tests. Run via check_task2.py --qc-tests."""
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from quality_control import (KEY, raw_run_qc, input_qc, reproducibility_qc,
                             robustness, robustness_grids, run, top_transcripts,
                             LOCAL_POSITIVES, LOCAL_DELTAS, assess_runs, dashboard_run_qc, verify_raw_roster, heya8_shared_qc)
from run_differential import EPS, candidate_mask
from task2_pipeline import build_differential_pair, read_prediction


class QualityControlTests(unittest.TestCase):
    def test_valid_raw_agrees_with_existing_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "A_rep1_run1.csv"
            path.write_text("transcript_id,transcript_position,score\nENST00001.1,1,0.5\nENST00002.2,2,0.1\n")
            original = read_prediction(path)
            q = raw_run_qc(path)
            self.assertEqual(q["n_unique_sites"], len(original))
            self.assertEqual(q["n_positive"], int(original.score.ge(.5).sum()))
            self.assertAlmostEqual(q["mean_score"], original.score.mean())
            self.assertEqual(q["input_warnings"], "")
            self.assertEqual(q["n_pipeline_retained_records"], 2)
            self.assertEqual(q["valid_record_pct"], 100)
            assessed = assess_runs(pd.DataFrame([dict(q, cell_line="A", run="A_rep1_run1")]), pd.DataFrame())
            self.assertTrue(assessed.iloc[0].complete_input_qc)
            self.assertEqual(assessed.iloc[0].qc_status, "No issues detected")

    def test_raw_validation_normalized_duplicates_and_bad_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "A_rep1_run1.csv"
            path.write_text("transcript_id,transcript_position,score\n"
                            "ENST00001.1,1,0.5\nENST00001.2,1,0.7\n"
                            ",2,\nENST2,,inf\nENST3,3,-0.1\n"
                            "ENST4,bad,1.1\nENST5,4,not-a-score\nENST6,4.5,0.2\n")
            q = raw_run_qc(path)
            self.assertEqual(q["n_records"], 8)
            self.assertEqual(q["n_unique_sites"], 3)
            self.assertEqual(q["duplicate_site_records"], 1)
            self.assertEqual(q["missing_transcript_ids"], 1)
            self.assertEqual(q["missing_positions"], 1)
            self.assertEqual(q["invalid_positions"], 2)
            self.assertEqual(q["missing_scores"], 1)
            self.assertEqual(q["invalid_scores"], 1)
            self.assertEqual(q["nonfinite_scores"], 3)
            self.assertEqual(q["out_of_range_scores"], 2)
            self.assertTrue(pd.isna(q["score_q95"]))
            self.assertEqual(q["n_valid_records"], 0)
            self.assertTrue(pd.isna(q["positive_fraction"]))
            self.assertEqual(q["n_pipeline_retained_records"], 0)

    def test_cached_availability_does_not_become_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp)
            pd.DataFrame([dict(cell_line="A", run="A_rep1_run1", n_sites=10,
                               mean_score=.2, median_score=.1, score_sd=.3,
                               **{"positive_0.5": 2})]).to_csv(tables / "prediction_overview.csv", index=False)
            q = input_qc(tables)
            self.assertTrue(pd.isna(q.iloc[0].missing_scores))
            self.assertTrue(pd.isna(q.iloc[0].score_q95))
            self.assertTrue(pd.isna(q.iloc[0].n_records))
            self.assertEqual(q.iloc[0].n_cached_retained_records, 10)
            self.assertTrue(pd.isna(q.iloc[0].valid_record_pct))
            self.assertEqual(q.iloc[0].positive_fraction, .2)
            self.assertEqual(q.iloc[0].validation_status, "unavailable")
            rep = reproducibility_qc(tables, q)
            self.assertTrue(pd.isna(rep.iloc[0].mean_pearson))
            self.assertIn("unavailable", rep.iloc[0].warnings)

    def test_empty_and_malformed_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "empty.csv"
            p.write_text("transcript_id,transcript_position,score\n")
            q = raw_run_qc(p)
            self.assertEqual(q["n_records"], 0)
            self.assertTrue(pd.isna(q["positive_fraction"]))
            self.assertIn("empty", q["input_warnings"])
            (root / "A").mkdir()
            (root / "A" / "A_rep1_run1.csv").write_text("wrong,schema\n1,2\n")
            q = input_qc(root, root)
            self.assertEqual(q.iloc[0].validation_status, "unavailable")
            self.assertTrue(pd.isna(q.iloc[0].n_records))
            self.assertTrue(q.iloc[0].structural_validation_issue)
            assessed = assess_runs(q, pd.DataFrame())
            self.assertEqual(assessed.iloc[0].qc_status, "Review recommended")
            # Missing all scientific outputs still emits informative QC CSVs.
            run(root / "new-output")
            self.assertTrue((root / "new-output/tables/qc_overview.csv").exists())

    @staticmethod
    def universe():
        return pd.DataFrame({"transcript_id": ["B", "A", "C", "D"],
                             "transcript_position": [1, 1, 2, 3],
                             "higher_mean_score": [.8, .8, .5 - EPS / 2, .2],
                             "abs_delta_mean": [.6, .6, .3 - EPS / 2, .1],
                             "replicate_supported": [True, True, False, False],
                             "higher_cell": ["Hct116"] * 4})

    def test_selection_overlap_strict_and_ranking(self):
        d = self.universe()
        self.assertEqual(candidate_mask(d, .5, .3).sum(), 3)
        q = robustness(d, positives=[.5, .9], deltas=[.3, .5])
        base = q[q.is_baseline].iloc[0]
        self.assertEqual(base.n_selected, 3)
        self.assertEqual(base.n_strict, 2)
        self.assertAlmostEqual(base.strict_fraction, 2/3)
        self.assertEqual(base.baseline_retention, 1)
        self.assertEqual(base.baseline_retention_pct, 100)
        self.assertEqual(base.candidate_jaccard_vs_baseline, 1)
        narrower = q[(q.positive_threshold == .5) & (q.delta_threshold == .5)].iloc[0]
        self.assertAlmostEqual(narrower.candidate_jaccard_vs_baseline, 2/3)
        self.assertAlmostEqual(narrower.baseline_retention_pct, 200/3)
        self.assertEqual(top_transcripts(d), ["A", "B", "C", "D"])
        pd.testing.assert_frame_equal(q, robustness(d.sample(frac=1, random_state=42), positives=[.5, .9], deltas=[.3, .5]))
        empty = q[q.positive_threshold == .9].iloc[0]
        self.assertEqual(empty.n_selected, 0)
        self.assertEqual(empty.n_strict, 0)
        self.assertTrue(pd.isna(empty.strict_support_pct))
        self.assertTrue(pd.isna(empty.strict_fraction))
        self.assertEqual(empty.candidate_jaccard_vs_baseline, 0)
        self.assertTrue(robustness(d.iloc[:0]).n_selected.eq(0).all())
        self.assertTrue(robustness(d.iloc[:0]).strict_fraction.isna().all())
        self.assertTrue(robustness(d.iloc[:0]).baseline_retention.isna().all())
        self.assertTrue(robustness(d.iloc[:0]).candidate_jaccard_vs_baseline.isna().all())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            robustness(pd.concat([d, d.iloc[:1]]))

    def test_matched_universe_and_strict_range_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp)
            for cell, mean, lower, upper, counts in (
                ("Hct116", [.8, .8, .8], [.5, .51, .7], [.9, .9, .9], [2, 2, 1]),
                ("K562", [.2, .2, .2], [.1, .1, .1], [.5, .5, .3], [2, 2, 2]),
            ):
                pd.DataFrame({"transcript_id": ["A", "B", "C"], "transcript_position": [1, 2, 3],
                              "mean_score": mean, "min_score": lower, "max_score": upper,
                              "std_score": [.1]*3, "n_detected_runs": counts,
                              "n_positive_runs": [0]*3}).to_parquet(tables / f"consensus_{cell}.parquet")
            d, selected, strict = build_differential_pair(tables, "Hct116", "K562")
            self.assertEqual(set(d.transcript_id), {"A", "B"})
            self.assertEqual(len(selected), 2)
            self.assertEqual(strict.transcript_id.tolist(), ["B"])
            q = robustness(d)
            self.assertEqual(q[q.is_baseline].iloc[0].n_strict, 1)

    def test_two_run_limit_and_technical_label_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp)
            runs = pd.DataFrame({"cell_line": ["A", "A"], "run": ["A_rep1_run1", "A_rep1_run2"]})
            pd.DataFrame([dict(cell_line="A", run_a="A_rep1_run1", run_b="A_rep1_run2",
                               fraction_a=.5, fraction_b=.7, pearson=.6, spearman=.5,
                               n_shared=5, **{"jaccard_0.5": .2})]).to_csv(tables / "replicate_pairwise.csv", index=False)
            q = reproducibility_qc(tables, runs)
            self.assertTrue(q.same_replicate_label_pairs.eq(1).all())
            self.assertTrue(q.warnings.str.contains("Only two runs").all())
            self.assertTrue(q.warnings.str.contains("technical runs").all())
            self.assertTrue(q.cell_union_sites.isna().all())

    def test_valid_population_positive_rate_and_raw_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "A_rep1_run1.csv"
            rows = ["ENST1,1,0.5", "ENST2,2,0.1", "ENST3,3,0.9",
                    "ENST4,4,1.2", "ENST5,5,inf", "ENST6,6,0.8", "ENST6,6,0.3", ",7,0.8"]
            path.write_text("transcript_id,transcript_position,score\n" + "\n".join(rows))
            q = raw_run_qc(path)
            self.assertEqual(q["n_records"], 8)
            self.assertEqual(q["n_valid_records"], 3)
            self.assertEqual(q["valid_record_pct"], 37.5)
            self.assertEqual(q["n_positive"], 2)
            self.assertAlmostEqual(q["positive_fraction"], 2/3)
            self.assertEqual(q["duplicate_member_records"], 2)
            self.assertEqual(q["n_pipeline_retained_records"], 0)
            self.assertIn("rejects entire file", q["pipeline_input_handling"])
            path.write_text("transcript_id,transcript_position,score\n" + "\n".join(reversed(rows)))
            reordered = raw_run_qc(path)
            for key in q:
                if key == "pipeline_input_handling":
                    continue  # Loader's first reported error can depend on input order.
                if isinstance(q[key], (float, np.floating)):
                    np.testing.assert_allclose(q[key], reordered[key], rtol=1e-6, equal_nan=True)
                else:
                    self.assertEqual(q[key], reordered[key])

    def test_warning_fences_and_two_run_ambiguity(self):
        inputs = pd.DataFrame({"cell_line": ["A"]*5, "run": [f"A_rep{i}_run1" for i in range(5)],
                               "n_predicted_sites": [100, 101, 102, 103, 10000],
                               "positive_rate_pct": [1, 1.01, 1.02, 1.03, 5],
                               "mean_score": [.1, .101, .102, .103, .5],
                               "validation_status": ["unavailable"]*5, "input_warnings": [""]*5})
        rep = pd.DataFrame({"cell_line": ["A"]*5, "run": inputs.run,
                            "mean_pearson": [.6, .61, .62, .63, .1],
                            "mean_fraction_of_run_sites_shared": [.7, .71, .72, .73, .1]})
        q = assess_runs(inputs, rep)
        self.assertEqual(q.has_descriptive_warning.sum(), 1)
        warning = q.iloc[-1].descriptive_warnings
        for label in ("site count", "positive rate", "mean score", "Pearson", "site overlap"):
            self.assertIn(label, warning)
        self.assertEqual(q.iloc[-1].qc_status, "Review recommended")
        pd.testing.assert_frame_equal(q.sort_values("run").reset_index(drop=True),
            assess_runs(inputs.sample(frac=1, random_state=1), rep.sample(frac=1, random_state=2)).sort_values("run").reset_index(drop=True))
        pair = assess_runs(inputs.iloc[[0, 4]], rep.iloc[[0, 4]])
        self.assertFalse(pair.has_descriptive_warning.any())
        self.assertTrue(pair.qc_context.str.contains("attribution ambiguous").all())

    def test_partial_dashboard_data_and_cached_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp)
            pd.DataFrame([dict(cell_line="A", run="A_rep1_run1", n_sites=10,
                mean_score=.2, median_score=.1, score_sd=.3, **{"positive_0.5": 2})]).to_csv(tables / "prediction_overview.csv", index=False)
            q = dashboard_run_qc(tables)
            self.assertEqual(q.iloc[0].n_predicted_sites, 10)
            self.assertFalse(q.iloc[0].complete_input_qc)
            pd.DataFrame([dict(cell_line="A", run="A_rep1_run1", validation_status="checked", missing_scores=0, n_positive=9)]).to_csv(tables / "qc_run_summary.csv", index=False)
            partial = dashboard_run_qc(tables)
            self.assertFalse(partial.iloc[0].complete_input_qc)
            self.assertTrue(pd.isna(partial.iloc[0].duplicate_site_records))
            self.assertEqual(partial.iloc[0].positive_rate_pct, 20)
            self.assertEqual(partial.iloc[0].n_positive, 2)  # Incomplete numerator/denominator cannot replace a cached population.
            self.assertEqual(partial.iloc[0].qc_status, "Input QC unavailable")

    def test_dashboard_difference_cutoff_survives_navigation(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parent
        app = AppTest.from_file(str(root / "dashboard.py"), default_timeout=60).run()

        def navigate(page):
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])

        def candidate_count():
            return next(m.value for m in app.metric if m.label == "Matching candidate sites")

        self.assertEqual(len(app.sidebar.slider), 0)
        navigate("Differential Atlas")
        self.assertEqual(app.sidebar.select_slider[0].value, .5)
        self.assertEqual(app.sidebar.slider[0].value, .3)
        self.assertEqual(candidate_count(), "2,736")
        app.sidebar.slider[0].set_value(.5).run()
        self.assertEqual(candidate_count(), "526")
        original_charts = [chart.proto.spec for chart in app.get("plotly_chart")]

        for hidden_page in ("Overview", "Reproducibility", "Cross-cell Similarity"):
            with self.subTest(page=hidden_page):
                navigate(hidden_page)
                self.assertEqual(len(app.sidebar.slider), 0)
                # Cleanup must also survive further reruns while the widget is absent.
                app.run()
                navigate("Differential Atlas")
                self.assertEqual(app.sidebar.slider[0].value, .5)
                self.assertEqual(candidate_count(), "526")
                self.assertEqual([chart.proto.spec for chart in app.get("plotly_chart")], original_charts)

        navigate("Transcript Explorer")
        self.assertEqual(app.sidebar.slider[0].value, .5)
        app.text_input[0].set_value("ENST00000299238").run()
        atlas = pd.read_parquet(root / "outputs/tables/differential_atlas/positive_0.5/differential_site_atlas.parquet")
        for cutoff in (.5, .45):
            app.sidebar.slider[0].set_value(cutoff).run()
            expected = atlas[(atlas.transcript_id == "ENST00000299238")
                             & (atlas.pair == "Hct116_vs_K562")
                             & (atlas.abs_delta_mean >= cutoff - EPS)]
            self.assertTrue(any(f"{len(expected)} Hct116–K562 differential candidates" in m.value for m in app.markdown))
        navigate("Overview")
        navigate("Transcript Explorer")
        self.assertEqual(app.sidebar.slider[0].value, .45)
        navigate("Differential Atlas")
        self.assertEqual(app.sidebar.slider[0].value, .45)
        expected = atlas[(atlas.pair == "Hct116_vs_K562") & (atlas.abs_delta_mean >= .45 - EPS)]
        self.assertEqual(candidate_count(), f"{len(expected):,}")

    def test_streamlit_scientific_pages_without_qc_interface(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parent
        note = "Data quality: All 23 prediction runs (28.67 million records) were checked for missing values, duplicate sites, and invalid scores; none were detected."
        def render(path):
            app = AppTest.from_file(str(path), default_timeout=60).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            return app
        disclaimer = "Results represent predicted methylation scores, not experimentally validated methylation status."
        def no_qc_interface(app, overview=False):
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(sum(c.value == disclaimer for c in app.sidebar.caption), 1)
            self.assertEqual(sum(c.value == disclaimer for c in app.caption), 1)
            self.assertFalse(any("Not experimental differential methylation testing" in c.value for c in app.caption))
            self.assertFalse(any(m.value.startswith("**Finding:**") for m in app.markdown))
            self.assertEqual(sum(c.value == note for c in app.caption), int(overview))
            forbidden = ("Complete input QC", "Raw-input QC unavailable", "Descriptive QC warnings", "Raw inputs audited", "Runs with QC warnings")
            self.assertFalse(any(m.label in forbidden for m in app.metric))
            self.assertFalse(any("QC" in e.label or "Robustness" in e.label or "diagnostic" in e.label.lower() for e in app.expander))
            self.assertFalse(any(w.key == "qc_run" for w in app.selectbox))
            self.assertFalse(any("qc_" in str(button.key) or "quality_control" in str(button.key) for button in app.get("download_button")))
        app = render(root / "dashboard.py")
        no_qc_interface(app, overview=True)
        self.assertEqual(len(app.get("plotly_chart")), 3)
        self.assertEqual(len(app.get("image")), 0)
        figure = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(figure["layout"]["font"]["size"], 14)
        self.assertEqual(figure["layout"]["title"]["font"]["size"], 19)
        self.assertEqual(figure["layout"]["xaxis"]["tickfont"]["size"], 13)
        self.assertEqual(app.dataframe[0].value.columns.tolist(), ["Cell line", "Run", "Predicted sites", "Positive predictions", "Positive rate (%)"])
        baseline_counts = app.dataframe[0].value["Positive predictions"].tolist()
        app.sidebar.select_slider[0].set_value(.9).run()
        no_qc_interface(app, overview=True)
        self.assertLess(sum(app.dataframe[0].value["Positive predictions"]), sum(baseline_counts))
        counts = app.dataframe[0].value["Positive predictions"].tolist()
        self.assertEqual(len(app.sidebar.slider), 0)
        self.assertEqual(app.dataframe[0].value["Positive predictions"].tolist(), counts)
        app.sidebar.select_slider[0].set_value(.5).run()
        for page, charts in (("Reproducibility",2),("Cross-cell Similarity",2),("Differential Atlas",3)):
            app.sidebar.radio[0].set_value(page).run()
            no_qc_interface(app)
            self.assertEqual(len(app.get("plotly_chart")), charts)
            expected_result = {"Reproducibility":"0.590", "Cross-cell Similarity":"0.794", "Differential Atlas":"2,736 of 810,786"}[page]
            self.assertTrue(any(expected_result in m.value for m in app.markdown))
        self.assertEqual(len(app.get("image")), 1)  # Scientific Figure 4 only.
        self.assertTrue(any("2,736 of 810,786" in m.value and "2,230" in m.value for m in app.markdown))
        self.assertEqual({m.label:m.value for m in app.metric}["Matching candidate sites"], "2,736")
        next(r for r in app.radio if r.label == "Candidate set").set_value("Strict").run()
        self.assertEqual({m.label:m.value for m in app.metric}["Matching candidate sites"], "2,230")
        next(r for r in app.radio if r.label == "Candidate set").set_value("Selected").run()
        app.sidebar.select_slider[0].set_value(.7).run()
        app.sidebar.slider[0].set_value(.4).run()
        self.assertEqual({m.label:m.value for m in app.metric}["Matching candidate sites"], "294")
        self.assertTrue(any("2,736 of 810,786" in m.value for m in app.markdown))
        app.sidebar.select_slider[0].set_value(.9).run()
        app.sidebar.slider[0].set_value(.9).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("No candidates match" in i.value for i in app.info))
        app.sidebar.select_slider[0].set_value(.5).run()
        app.sidebar.slider[0].set_value(.3).run()
        app.sidebar.radio[0].set_value("Transcript Explorer").run()
        app.text_input[0].set_value("ENST00000299238").run()
        no_qc_interface(app)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertTrue(any("15 Hct116–K562 differential candidates" in m.value for m in app.markdown))
        # Captions and plots refer to the same stored candidates and scientific values.
        spec = app.get("plotly_chart")[0].proto.spec
        self.assertIn("Hct116–K562 differential site", [t.get("name") for t in json.loads(spec)["data"]])
        app.checkbox[0].set_value(True).run()
        self.assertFalse(app.exception)
        app.selectbox[0].set_value("Hct116").run()
        self.assertFalse(app.exception)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            shutil.copy(root / "dashboard.py", dest / "dashboard.py")
            shutil.copy(root / "dashboard.css", dest / "dashboard.css")
            tables = dest / "outputs/tables"; tables.mkdir(parents=True)
            figures = dest / "outputs/figures"; figures.mkdir()
            figure = root / "outputs/figures/fig4_differential_Hct116_vs_K562.png"
            (figures / figure.name).symlink_to(figure)
            for path in (root / "outputs/tables").iterdir():
                if not path.name.startswith("qc_"):
                    (tables / path.name).symlink_to(path, target_is_directory=path.is_dir())
            # No QC module, QC CSVs, S2/S3 or external raw directory in this app copy.
            with patch.dict("os.environ", {"TASK2_RAW_DIR":"unavailable-external-source"}):
                missing = render(dest / "dashboard.py")
                no_qc_interface(missing, overview=True)
                self.assertEqual(missing.dataframe[0].value["Positive predictions"].tolist(), baseline_counts)
                pd.DataFrame([dict(run="partial", missing_scores=0)]).to_csv(tables/"qc_run_summary.csv",index=False)
                partial = render(dest / "dashboard.py")
                no_qc_interface(partial, overview=True)
                self.assertEqual(partial.dataframe[0].value["Positive predictions"].tolist(), baseline_counts)
                for page in ("Reproducibility","Cross-cell Similarity","Differential Atlas","Transcript Explorer"):
                    partial.sidebar.radio[0].set_value(page).run()
                    no_qc_interface(partial)
        source = (root/"dashboard.py").read_text()
        for forbidden in ("from quality_control", "qc_run_summary", "figS2_qc", "figS3_run_level", "qc_heya8"):
            self.assertNotIn(forbidden, source)

    def test_real_run_qc_and_preserved_warning(self):
        tables = Path(__file__).resolve().parent / "outputs/tables"
        q = pd.read_csv(tables / "qc_run_summary.csv")
        cached = pd.read_csv(tables / "prediction_overview.csv")
        self.assertEqual(len(q), 23)
        self.assertTrue(q.complete_input_qc.all())
        for key in ("n_records", "n_valid_records", "valid_record_pct", "missing_scores", "duplicate_site_records", "score_q95"):
            self.assertTrue(q[key].notna().all())
        self.assertTrue(q.valid_record_pct.eq(100).all())
        self.assertTrue(q.cached_statistics_agree.all())
        for key in ("missing_scores", "duplicate_site_records", "nonfinite_scores", "out_of_range_scores"):
            self.assertTrue(q[key].eq(0).all())
        self.assertEqual(set(q.loc[q.has_descriptive_warning, "run"]), {"HEYA8_rep1_run1", "HEYA8_rep1_run2"})
        joined = q.merge(cached, on=["run", "cell_line"])
        self.assertTrue(joined.n_positive.eq(joined["positive_0.5"]).all())
        np.testing.assert_allclose(joined.positive_fraction, joined["positive_0.5"] / joined.n_sites)

    def test_cross_chunk_duplicates_exact_quantiles_and_safe_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "A_rep1_run1.csv"
            p.write_text("transcript_id,transcript_position,score\nENST1.1,1,0.1\nENST2,2,0.5\nENST3,3,0.9\nENST1.2,1,0.7\n")
            small, large = raw_run_qc(p, chunksize=2), raw_run_qc(p, chunksize=100)
            self.assertEqual(small,large)
            self.assertEqual(small["duplicate_site_records"],1)
            self.assertEqual(small["duplicate_member_records"],2)
            self.assertEqual(small["n_valid_scored_sites"],2)
            expected = pd.Series([.5,.9],dtype="float32")
            self.assertEqual(small["score_q95"],expected.quantile(.95))
            self.assertNotIn(tmp,str(small))
            self.assertFalse(Path(small["source_path"]).is_absolute())

    def test_dashboard_deployment_never_audits_raw_data(self):
        tables = Path(__file__).resolve().parent / "outputs/tables"
        with patch.dict("os.environ", {"TASK2_RAW_DIR":"unavailable-external-source"}), \
             patch("quality_control.discover",side_effect=AssertionError("Raw discovery at startup")), \
             patch("quality_control.raw_run_qc",side_effect=AssertionError("Raw QC at startup")), \
             patch("quality_control.assess_runs",side_effect=AssertionError("QC recomputed despite ready artifacts")), \
             patch("quality_control.reproducibility_qc",side_effect=AssertionError("Pair statistics recomputed")):
            q = dashboard_run_qc(tables)
            self.assertEqual(len(q),23)
            self.assertTrue(q.complete_input_qc.all())

    def test_heya8_population_comparison_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tables=root/"tables"; tables.mkdir()
            a=root/"HEYA8_rep1_run1.csv"; b=root/"HEYA8_rep1_run2.csv"
            a.write_text("transcript_id,transcript_position,score\nENST1,1,.8\nENST2,2,.2\nENST3,3,.4\nENST4,4,.1\n")
            b.write_text("transcript_id,transcript_position,score\nENST1,1,.6\nENST2,2,.1\nENST3,3,.1\nENST5,5,0\n")
            pair=heya8_shared_qc(a,b,tables)
            self.assertEqual(pair["n_shared"],3)
            self.assertEqual(pair["fraction_a"],.75)
            self.assertEqual(pair["fraction_b"],.75)
            self.assertAlmostEqual(pair["positive_rate_shared_a_pct"],100/3,places=4)
            self.assertAlmostEqual(pair["mean_paired_score_difference"],.2,places=6)
            distributions=pd.read_csv(tables/"qc_heya8_score_distributions.csv")
            self.assertEqual(len(distributions),6)
            self.assertEqual(set(distributions.population),{"all_sites","shared_sites","run_only_sites"})

    def test_external_roster_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"A").mkdir(); (root/"tables").mkdir()
            (root/"A/A_rep1_run1.csv").write_text("transcript_id,transcript_position,score\nENST1,1,.5\n")
            pd.DataFrame([dict(cell_line="A",run="A_rep2_run1")]).to_csv(root/"tables/prediction_overview.csv",index=False)
            with self.assertRaisesRegex(ValueError,"roster mismatch"):
                verify_raw_roster(root/"tables",root)

    def test_real_baseline_and_grid(self):
        root = Path(__file__).resolve().parent / "outputs/tables"
        q = pd.read_csv(root / "qc_differential_robustness.csv")
        base = q[q.is_baseline].iloc[0]
        selected = pd.read_csv(root / "differential_selected_Hct116_vs_K562.csv")
        strict = pd.read_csv(root / "differential_strict_Hct116_vs_K562.csv")
        self.assertEqual(base.n_selected, len(selected))
        self.assertEqual(base.n_strict, len(strict))
        self.assertEqual(base.n_matched_sites, 810786)
        self.assertEqual(base.n_selected, 2736)
        self.assertEqual(base.n_strict, 2230)
        self.assertAlmostEqual(base.strict_fraction, 2230/2736)
        self.assertEqual(base.candidate_jaccard_vs_baseline, 1)
        self.assertEqual(base.baseline_retention, 1)
        self.assertEqual(q.n_matched_sites.nunique(), 1)
        self.assertEqual(len(q), 43)
        self.assertFalse(q.duplicated(["positive_threshold", "delta_threshold"]).any())
        local = q[q.grid_id.str.contains("local", regex=False)]
        self.assertEqual(len(local), 25)
        self.assertEqual(set(local.positive_threshold), set(LOCAL_POSITIVES))
        self.assertEqual(set(local.delta_threshold), set(LOCAL_DELTAS))
        self.assertEqual(q.grid_id.str.contains("broad", regex=False).sum(), 20)
        np.testing.assert_allclose(q.strict_fraction, q.n_strict / q.n_selected)
        np.testing.assert_allclose(q.strict_support_pct, q.strict_fraction * 100)
        np.testing.assert_allclose(q.baseline_retention_pct, q.baseline_retention * 100)
        self.assertTrue(q.candidate_jaccard_vs_baseline.between(0, 1).all())
        self.assertTrue((q.n_strict <= q.n_selected).all())
        overview = pd.read_csv(root / "qc_overview.csv").set_index("metric")
        for name in ("selected", "strict"):
            self.assertEqual(str(overview.loc[f"baseline_{name}_regression_matches", "value"]), "True")
        # Raising either threshold cannot add candidates.
        for col, groups in (("positive_threshold", "delta_threshold"), ("delta_threshold", "positive_threshold")):
            for _, group in q.groupby(groups):
                self.assertTrue(group.sort_values(col).n_selected.diff().dropna().le(0).all())

    def test_explicit_candidate_sets_jaccard_and_retention(self):
        # Baseline = {A,B}; relaxed = {A,B,C}; crossed = {B,D}.
        d = pd.DataFrame({"transcript_id": list("ABCD"), "transcript_position": [1]*4,
                          "higher_mean_score": [.55, .70, .45, .65],
                          "abs_delta_mean": [.35, .32, .40, .25],
                          "replicate_supported": [True, False, True, True],
                          "higher_cell": ["Hct116"]*4})
        q = robustness(d, [.4, .5, .6], [.2, .3])
        for pos, delta, expected in ((.5, .3, {"A", "B"}), (.4, .3, {"A", "B", "C"}), (.6, .2, {"B", "D"})):
            actual = set(d.loc[candidate_mask(d, pos, delta), "transcript_id"])
            self.assertEqual(actual, expected)
            row = q[(q.positive_threshold == pos) & (q.delta_threshold == delta)].iloc[0]
            baseline = {"A", "B"}
            self.assertEqual(row.n_selected, len(expected))
            self.assertAlmostEqual(row.candidate_jaccard_vs_baseline, len(expected & baseline) / len(expected | baseline))
            self.assertAlmostEqual(row.baseline_retention, len(expected & baseline) / len(baseline))
            self.assertAlmostEqual(row.strict_fraction, len(expected & {"A", "C", "D"}) / len(expected))

    def test_grid_deduplication_and_row_invariance(self):
        d = self.universe()
        q = robustness_grids(d)
        shuffled = d.sample(frac=1, random_state=9).reset_index(drop=True)
        pd.testing.assert_frame_equal(q, robustness_grids(shuffled))
        self.assertEqual(len(q), 43)
        self.assertEqual(q[q.is_baseline].iloc[0].grid_id, "local;broad")
        for pos in LOCAL_POSITIVES:
            for delta in LOCAL_DELTAS:
                before = set(map(tuple, d.loc[candidate_mask(d, pos, delta), KEY].to_numpy()))
                after = set(map(tuple, shuffled.loc[candidate_mask(shuffled, pos, delta), KEY].to_numpy()))
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
