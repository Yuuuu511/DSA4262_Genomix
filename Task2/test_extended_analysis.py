"""Focused tests for fixed-universe sharing and site-level variability summaries."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from run_extended_analysis import (CELLS, KEY, common_sites, sharing, variability,
    mean_pair_differences, ecdf_grid, analysis_a)
from task2_pipeline import build_differential_pair


class ExtendedAnalysisTests(unittest.TestCase):
    def test_common_universe_missing_is_not_negative(self):
        d=pd.DataFrame({"transcript_id":["A","B","C","D"],"transcript_position":[1]*4,
                        **{c:[.1,.8,.6,.1] for c in CELLS}})
        d.loc[1,"K562"]=np.nan
        d.loc[2,"H9"]=np.inf
        d.loc[3,"MCF7"]=-.1
        result=common_sites(d)
        self.assertEqual(result.transcript_id.tolist(),["A"])
        distribution,_,_=sharing(result)
        self.assertEqual(distribution.iloc[0].n_sites,1)
        with self.assertRaises(ValueError):sharing(d)
        with self.assertRaises(ValueError):common_sites(pd.concat([d,d.iloc[:1]]))

    def test_seven_cell_eligibility_rejects_one_run_or_missing_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/"new";out.mkdir()
            wide=pd.DataFrame(dict(transcript_id=["A","B","C"],transcript_position=[1]*3,
                **{c:[.5,.5,.5] for c in CELLS}))
            wide.loc[1,"MCF7"]=np.nan
            wide.loc[2,"A549"]=np.nan  # One-run site is absent from canonical wide cache.
            wide.to_parquet(root/"cross_cell_scores.parquet")
            for cell in CELLS:
                raw=pd.DataFrame(dict(transcript_id=["A","B","C"],transcript_position=[1]*3,
                    mean_score=[.5]*3,min_score=[.5]*3,max_score=[.5]*3,std_score=[0]*3,
                    n_detected_runs=[2,2,1 if cell=="A549" else 2],n_total_runs=[2]*3,n_positive_runs=[2]*3))
                if cell=="MCF7":raw=raw[raw.transcript_id!="B"]
                raw.to_parquet(root/f"consensus_{cell}.parquet")
            pd.DataFrame([dict(n_cell_lines_observed=7,n_cell_lines_positive=7,n_sites=1),
                          dict(n_cell_lines_observed=6,n_cell_lines_positive=6,n_sites=2)]).to_csv(root/"site_sharing.csv",index=False)
            summary,_,_,_=analysis_a(root,out)
            self.assertTrue(summary.n_eligible.eq(1).all())
            # An inconsistent cache must fail, never silently accept one-run data.
            wide.loc[2,"A549"]=.5;wide.to_parquet(root/"cross_cell_scores.parquet")
            with self.assertRaisesRegex(ValueError,"eligibility"):
                analysis_a(root,out)

    def test_positive_boundaries_and_seven_cell_sharing(self):
        below=np.nextafter(np.float32(.5),np.float32(0))
        d=pd.DataFrame({"transcript_id":["none","unique","two","all"],"transcript_position":[1]*4,
                        **{c:np.array([below,.1,.1,.5],dtype=np.float32) for c in CELLS}})
        d.loc[1,"Hct116"]=.5
        d.loc[2,["A549","K562"]]=.5
        result,n,pos=sharing(common_sites(d))
        counts=result.set_index("n_positive_cells").n_sites
        self.assertEqual(counts.loc[0],1)
        self.assertEqual(counts.loc[1],1)
        self.assertEqual(counts.loc[2],1)
        self.assertEqual(counts.loc[7],1)
        self.assertEqual(result.n_eligible.unique().tolist(),[4])
        self.assertAlmostEqual(result.query('n_positive_cells == 1').iloc[0].pct_positive_union,100/3)
        self.assertTrue(pd.isna(result.iloc[0].pct_positive_union))
        reordered=common_sites(d.sample(frac=1,random_state=7))
        pd.testing.assert_frame_equal(result,sharing(reordered)[0])
        self.assertEqual(int(sharing(common_sites(d),.55)[1].gt(0).sum()),0)

    def test_ranges_minimum_runs_and_strict_touching_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for c,mean,lo,hi,counts in (
                ("Hct116",[.8,.7,.8],[.6,.5,.7],[.9,.8,.9],[2,2,1]),
                ("K562",[.2,.2,.2],[.1,.1,.1],[.5,.5,.3],[2,2,2])):
                pd.DataFrame(dict(transcript_id=["B","A","C"],transcript_position=[1]*3,
                    mean_score=mean,min_score=lo,max_score=hi,std_score=[.1]*3,
                    n_detected_runs=counts,n_positive_runs=[0]*3)).to_parquet(root/f"consensus_{c}.parquet")
            matched,selected,strict=build_differential_pair(root,"Hct116","K562")
            d=variability(matched)
            self.assertEqual(d.transcript_id.tolist(),["A","B"])
            np.testing.assert_allclose(d.range_Hct116,[.3,.3])
            np.testing.assert_allclose(d.range_K562,[.4,.4])
            np.testing.assert_allclose(d.max_within_range,[.4,.4])
            self.assertEqual(strict.transcript_id.tolist(),["B"])
            self.assertEqual(d.replicate_supported.tolist(),[False,True])
            pd.testing.assert_frame_equal(d,variability(matched.sample(frac=1,random_state=1)))

    def test_site_level_pair_aggregation_without_pseudoreplication(self):
        # Site 1 has 3 pairs; site 2 only 1. Both still contribute one value.
        a=np.array([[0,.3,.9],[.2,.6,np.nan],[np.nan,.1,np.nan]])
        b=np.array([[.1,.2,np.nan],[0,.3,.4],[.1,np.nan,np.nan]])
        within,counts=mean_pair_differences(a)
        np.testing.assert_allclose(within[:2],[.6,.4])
        self.assertTrue(np.isnan(within[2]))
        self.assertEqual(counts.tolist(),[3,1,0])
        self.assertAlmostEqual(np.nanmean(within),.5)  # Not the pair-weighted .55.
        cross,ncross=mean_pair_differences(a,b)
        expected=[np.mean([abs(x-y) for x in row_a if np.isfinite(x) for y in row_b if np.isfinite(y)]) for row_a,row_b in zip(a,b)]
        np.testing.assert_allclose(cross,expected)
        self.assertEqual(ncross.tolist(),[6,6,1])
        self.assertEqual(len(cross),3)
        perm=[2,0,1]
        np.testing.assert_allclose(mean_pair_differences(a[perm],b[perm])[0],cross[perm])
        # Run-column order also cannot change the site-level definition.
        np.testing.assert_allclose(mean_pair_differences(a[:,::-1])[0],within,equal_nan=True)
        with self.assertRaises(ValueError):mean_pair_differences(a,b[:2])

    def test_exact_ecdf_grid_and_empty_data(self):
        d=ecdf_grid([0,.1,.1,.9])
        self.assertEqual(d.loc[np.isclose(d.score_difference,.1),"ecdf"].iloc[0],.75)
        self.assertEqual(d.iloc[-1].ecdf,1)
        pd.testing.assert_frame_equal(d,ecdf_grid([.1,.9,0,.1]))
        self.assertTrue(ecdf_grid([]).ecdf.isna().all())

    def test_real_results_and_baseline(self):
        root=Path(__file__).resolve().parent/"outputs/tables/extended_analysis"
        a=pd.read_csv(root/"a_sharing_distribution.csv")
        baseline=a[a.threshold.eq(.5)]
        self.assertEqual(int(baseline.n_sites.sum()),440569)
        self.assertEqual(int(baseline.query('n_positive_cells == 1').iloc[0].n_sites),4059)
        b=pd.read_csv(root/"b_variability_summary.csv").set_index("group")
        self.assertEqual(int(b.loc["all_matched","n_sites"]),810786)
        self.assertEqual(int(b.loc["baseline_candidates","n_sites"]),2736)
        self.assertEqual(int(b.loc["baseline_candidates","n_strict"]),2230)
        candidates=pd.read_csv(root/"b_baseline_candidate_variability.csv")
        self.assertFalse(candidates.duplicated(KEY).any())
        self.assertTrue((candidates.n_detected_runs_Hct116>=2).all())
        self.assertTrue((candidates.n_detected_runs_K562>=2).all())
        if "n_cross_run_pairs" in candidates:
            self.assertTrue((candidates.n_cross_run_pairs==candidates.n_detected_runs_Hct116*candidates.n_detected_runs_K562).all())


if __name__=="__main__":
    unittest.main()
