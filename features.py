"""Streaming site-level features; no labels, IDs or file offsets are predictors."""
import gzip,json
import numpy as np

def sites(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            try:
                for tid,positions in json.loads(line).items():
                    for pos,motifs in positions.items():
                        if len(motifs)!=1: raise ValueError('Expected one sequence per site')
                        seq,reads=next(iter(motifs.items()))
                        yield tid,str(pos),seq,reads
            except Exception as e: raise ValueError(f'{path}, line {line_no}: {e}') from e

def extract(seq,reads):
    a=np.asarray(reads,dtype=float)
    if a.ndim!=2 or a.shape[1]!=9 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Reads must be a nonempty finite N x 9 matrix')
    if len(seq)!=7 or any(b not in 'ACGT' for b in seq): raise ValueError('Expected seven DNA bases')
    if (a[:,[0,1,3,4,6,7]]<0).any(): raise ValueError('Negative dwell time or signal SD')
    a[:,[0,3,6]]=np.log1p(a[:,[0,3,6]])
    # Quantiles retain heterogeneity expected from mixtures of modified reads.
    q=np.quantile(a,[.1,.25,.5,.75,.9],axis=0)
    return np.concatenate([a.mean(0),a.std(0),q.ravel(),q[3]-q[1],
        [float(b==v) for b in seq for v in 'ACGT']]).astype('float32')
