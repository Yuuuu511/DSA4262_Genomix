"""Small contract tests for feature semantics, malformed inputs and exact coverage."""
import gzip,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from features import site_features,FEATURE_NAMES,load_features,read_labels
from validate_submission import validate

class PipelineContracts(unittest.TestCase):
    reads=[[.01,2,100,.02,3,110,.03,4,90],[.02,4,120,.01,1,100,.02,2,80]]
    def test_feature_semantics_and_permutation(self):
        x=site_features('AAGACCA',self.reads)
        np.testing.assert_allclose(x,site_features('AAGACCA',self.reads[::-1]))
        np.testing.assert_allclose(x[:9],np.mean(self.reads,axis=0))
        self.assertEqual(len(x),len(FEATURE_NAMES))
        self.assertEqual(x[FEATURE_NAMES.index('motif_AGACC')],1)
        self.assertEqual(x[FEATURE_NAMES.index('base0_A')],1)
    def test_bad_read_inputs_fail(self):
        for reads in [[],[[1]*8],[[float('nan')]*9]]:
            with self.assertRaises(ValueError):site_features('AAGACCA',reads)
    def test_key_alignment_and_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'test.json.gz'
            with gzip.open(data,'wt') as f:
                f.write(json.dumps({'tx_id_0':{'0':{'AAGACCA':self.reads}}})+'\n')
            meta,X=load_features(data)
            self.assertEqual(meta.transcript_position.iloc[0],0)
            labels=meta.assign(gene_id='g0',label=1);labels.to_csv(root/'labels.csv',index=False)
            self.assertEqual(read_labels(root/'labels.csv',meta).label.iloc[0],1)
            meta.assign(score=.8).to_csv(root/'p.csv',index=False)
            self.assertEqual(validate(data,root/'p.csv')['status'],'PASS')
            meta.assign(transcript_position=1,score=.8).to_csv(root/'p.csv',index=False)
            with self.assertRaises(ValueError):validate(data,root/'p.csv')
            labels.assign(transcript_position=1).to_csv(root/'labels.csv',index=False)
            with self.assertRaises(ValueError):read_labels(root/'labels.csv',meta)
    def test_example_prediction_matches_full_file(self):
        root=Path(__file__).resolve().parent/'example'
        p=pd.read_csv(root/'predictions.csv');expected=pd.read_csv(root/'expected_predictions.csv')
        pd.testing.assert_frame_equal(p[['transcript_id','transcript_position']],expected[['transcript_id','transcript_position']])
        np.testing.assert_allclose(p.score,expected.score,rtol=0,atol=1e-6)

if __name__=='__main__':unittest.main()
