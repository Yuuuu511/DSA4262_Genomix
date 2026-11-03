"""Extended Task 2 analyses; cached consensus is authoritative, raw CSVs read-only.

Run: python run_extended_analysis.py --output outputs --input-dir /path/to/external/predictions
Raw input is optional (or TASK2_RAW_DIR); individual cross-run differences require it.
No original artifacts are written. See extended_analysis/methods.json and interpretation.md.
"""
from pathlib import Path
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

from task2_pipeline import (KEY, MAIN_T, MIN_RUNS, build_differential_pair,
                            read_prediction, discover, setup_style, savefig, get_palette)
from run_differential import candidate_mask, BASE_DELTA, EPS

CELLS = ("A549", "H9", "HEYA8", "Hct116", "HepG2", "K562", "MCF7")
SENSITIVITY = (.45, .5, .55)


def ratio(n, d):
    return n/d if d else np.nan


def common_sites(wide, cells=CELLS):
    """Wide cache already requires >=MIN_RUNS within every observed cell.

    A missing/nonfinite cell score never becomes a negative call. Input IDs have
    already been normalized by the canonical loader, which removes ENST versions.
    """
    if wide.duplicated(KEY).any():
        raise ValueError("Duplicate canonical site keys in wide cache")
    values = wide[list(cells)]
    eligible = np.isfinite(values).all(axis=1) & values.ge(0).all(axis=1) & values.le(1).all(axis=1)
    return wide.loc[eligible, KEY + list(cells)].sort_values(KEY).reset_index(drop=True)


def sharing(d, threshold=MAIN_T, cells=CELLS):
    """Exact >= threshold, matching consensus/cross-cell positive calls (no EPS)."""
    if not np.isfinite(d[list(cells)]).all().all():
        raise ValueError("Sharing requires a complete finite comparison universe")
    positive = d[list(cells)].ge(threshold)
    number = positive.sum(axis=1)
    any_count = int(number.gt(0).sum())
    rows = [dict(threshold=threshold, n_positive_cells=k, n_sites=int(number.eq(k).sum()),
                 n_eligible=len(d), n_positive_union=any_count,
                 pct_eligible=100*ratio(int(number.eq(k).sum()),len(d)),
                 pct_positive_union=100*ratio(int(number.eq(k).sum()),any_count) if k else np.nan)
            for k in range(len(cells)+1)]
    return pd.DataFrame(rows), number, positive


def describe(values):
    s = pd.Series(np.asarray(values,dtype=float)).dropna()
    return dict(n_observed=len(s), mean=s.mean(), median=s.median(), q05=s.quantile(.05),
                q25=s.quantile(.25), q75=s.quantile(.75), q95=s.quantile(.95), q99=s.quantile(.99))


def analysis_a(tables, destination):
    wide = pd.read_parquet(tables/"cross_cell_scores.parquet",columns=KEY+list(CELLS))
    complete = common_sites(wide)
    universe_union = len(wide)
    distributions, summaries, cell_rows, details, context = [], [], [], [], []
    for t in SENSITIVITY:
        distribution, number, positive = sharing(complete,t)
        distributions.append(distribution)
        union = int(number.gt(0).sum())
        summaries.append(dict(threshold=t,n_eligible=len(complete),n_union_evaluable=universe_union,
            eligible_pct_union=100*ratio(len(complete),universe_union),n_all_negative=int(number.eq(0).sum()),
            n_positive_union=union,n_multi_cell_positive=int(number.ge(2).sum()),
            multi_cell_pct_positive_union=100*ratio(int(number.ge(2).sum()),union),
            n_all_seven_positive=int(number.eq(7).sum()),all_seven_pct_positive_union=100*ratio(int(number.eq(7).sum()),union),
            all_seven_pct_eligible=100*ratio(int(number.eq(7).sum()),len(complete)),
            n_exactly_one_positive=int(number.eq(1).sum()),exactly_one_pct_positive_union=100*ratio(int(number.eq(1).sum()),union),
            exactly_one_pct_eligible=100*ratio(int(number.eq(1).sum()),len(complete))))
    # Load each consensus once; only compact summaries of specific sites survive.
    for cell in CELLS:
        keep=KEY+["mean_score","min_score","max_score","std_score","n_detected_runs","n_total_runs","n_positive_runs"]
        cell_data=pd.read_parquet(tables/f"consensus_{cell}.parquet",columns=keep)
        cell_data=cell_data[cell_data.n_detected_runs>=MIN_RUNS]
        common=complete.merge(cell_data,on=KEY,how="left",validate="one_to_one")
        if common.n_detected_runs.isna().any():
            raise ValueError("Wide eligibility does not agree with consensus detection counts")
        if not np.allclose(common[cell],common.mean_score,rtol=0,atol=0):
            raise ValueError("Wide scores do not agree with consensus means")
        if not np.isfinite(common[keep[len(KEY):]]).all().all():
            raise ValueError("Missing consensus variability/support summaries")
        for t in SENSITIVITY:
            _, number, positive=sharing(complete,t)
            mask=number.eq(1)&positive[cell]
            specific=common.loc[mask].copy()
            positive_total=int(positive[cell].sum())
            all_observed=specific.min_score.ge(t)
            stats=describe(specific.mean_score)
            row=dict(threshold=t,cell_line=cell,n_eligible=len(complete),n_positive_common=positive_total,
                n_specific=len(specific),specific_pct_cell_positive=100*ratio(len(specific),positive_total),
                n_all_observed_runs_positive=int(all_observed.sum()),
                all_observed_positive_pct_specific=100*ratio(int(all_observed.sum()),len(specific)),
                n_full_run_coverage=int(specific.n_detected_runs.eq(specific.n_total_runs).sum()),
                full_run_coverage_pct_specific=100*ratio(int(specific.n_detected_runs.eq(specific.n_total_runs).sum()),len(specific)),
                median_n_detected_runs=specific.n_detected_runs.median(),
                median_observed_min_score=specific.min_score.median(),median_observed_sd=specific.std_score.median(),
                **{f"positive_mean_{k}":v for k,v in stats.items()})
            if t == MAIN_T:
                specific["specific_cell"]=cell
                specific["all_observed_runs_positive"]=all_observed
                details.append(specific[KEY+["specific_cell"]+list(CELLS)+["mean_score","min_score","max_score","std_score","n_detected_runs","n_total_runs","n_positive_runs","all_observed_runs_positive"]])
            cell_rows.append(row)
        context.append(dict(cell_line=cell,n_cell_evaluable=len(cell_data),n_common=len(complete),
            common_pct_cell_evaluable=100*ratio(len(complete),len(cell_data)),
            n_cell_positive=int(cell_data.mean_score.ge(MAIN_T).sum()),
            n_common_positive=int(common[cell].ge(MAIN_T).sum()),
            common_positive_pct_cell_positive=100*ratio(int(common[cell].ge(MAIN_T).sum()),int(cell_data.mean_score.ge(MAIN_T).sum()))))
    distribution=pd.concat(distributions,ignore_index=True)
    summary=pd.DataFrame(summaries)
    specific_table=pd.concat(details,ignore_index=True).sort_values(KEY+["specific_cell"]).reset_index(drop=True)
    # Threshold stability of the same cell-specific site identities, not only counts.
    stability=[]
    _,n0,p0=sharing(complete,MAIN_T)
    for t in SENSITIVITY:
        _,nt,pt=sharing(complete,t)
        for cell in CELLS:
            base=(n0.eq(1)&p0[cell]); current=(nt.eq(1)&pt[cell])
            stability.append(dict(threshold=t,cell_line=cell,n_baseline_specific=int(base.sum()),
                n_same_cell_specific_overlap=int((base&current).sum()),
                baseline_specific_retention_pct=100*ratio(int((base&current).sum()),int(base.sum())),
                specific_jaccard=ratio(int((base&current).sum()),int((base|current).sum()))))
    for name,frame in (("a_sharing_distribution",distribution),("a_universe_summary",summary),
                       ("a_cell_specific_summary",pd.DataFrame(cell_rows)),("a_cell_specific_sites",specific_table),
                       ("a_coverage_context",pd.DataFrame(context)),("a_specific_sensitivity",pd.DataFrame(stability))):
        frame.to_csv(destination/f"{name}.csv",index=False)
    # Regression against the existing sharing table, restricted to the same universe.
    previous=pd.read_csv(tables/"site_sharing.csv")
    previous=previous[previous.n_cell_lines_observed==len(CELLS)].set_index("n_cell_lines_positive").n_sites
    actual=distribution[distribution.threshold.eq(MAIN_T)].set_index("n_positive_cells").n_sites
    if not actual.eq(previous.reindex(actual.index,fill_value=0)).all():
        raise ValueError("Baseline sharing counts differ from existing site_sharing.csv")
    return summary, distribution, pd.DataFrame(cell_rows), specific_table


def variability(d):
    """Use actual observed-run extrema/SDs already cached by the pipeline."""
    keep=(d.n_detected_runs_Hct116>=MIN_RUNS)&(d.n_detected_runs_K562>=MIN_RUNS)
    out=d.loc[keep].sort_values(KEY).reset_index(drop=True).copy()
    for cell in ("Hct116","K562"):
        out[f"range_{cell}"]=out[f"max_score_{cell}"]-out[f"min_score_{cell}"]
        if out[f"range_{cell}"].lt(0).any():
            raise ValueError("Inverted observed score extrema")
    out["max_within_range"]=np.maximum(out.range_Hct116,out.range_K562)
    out["max_within_sd"]=np.maximum(out.std_score_Hct116,out.std_score_K562)
    out["D_gt_max_range"]=out.abs_delta_mean>out.max_within_range
    out["baseline_selected"]=candidate_mask(out,MAIN_T,BASE_DELTA)
    return out


def mean_pair_differences(a, b=None):
    """Each row is a site. Average valid absolute pair differences within that site.

    Self comparisons use unordered distinct run pairs; cross-cell comparisons use
    every valid Hct116/K562 run combination. No site gets extra distribution weight
    merely because it has more observed runs. Pair counts are diagnostic only.
    """
    a=np.asarray(a,dtype=float)
    b=None if b is None else np.asarray(b,dtype=float)
    if b is not None and len(a)!=len(b):
        raise ValueError("Run matrices must have the same site universe")
    pairs=itertools.combinations(range(a.shape[1]),2) if b is None else itertools.product(range(a.shape[1]),range(b.shape[1]))
    total=np.zeros(len(a),dtype=float); counts=np.zeros(len(a),dtype=np.int16)
    for i,j in pairs:
        x=a[:,i]; y=a[:,j] if b is None else b[:,j]
        observed=np.isfinite(x)&np.isfinite(y)
        total[observed]+=np.abs(x[observed]-y[observed]); counts[observed]+=1
    return np.divide(total,counts,out=np.full(len(a),np.nan),where=counts>0),counts


def raw_run_matrices(raw_root, tables, universe):
    """Read each of the six focus CSVs individually; retain only aligned score arrays."""
    files=discover(raw_root)
    roster=pd.read_csv(tables/"prediction_overview.csv")
    index=pd.MultiIndex.from_frame(universe[KEY])
    matrices={}
    audit=[]
    for cell in ("Hct116","K562"):
        paths=sorted(files.get(cell,[]))
        expected=set(roster.loc[roster.cell_line.eq(cell),"run"])
        if {p.stem for p in paths}!=expected:
            raise ValueError(f"Focus raw roster mismatch for {cell}")
        matrix=np.full((len(universe),len(paths)),np.nan,dtype=np.float32)
        for j,path in enumerate(paths):
            print("Reading focus run",path.stem,flush=True)
            d=read_prediction(path)
            matrix[:,j]=d.set_index(KEY).score.reindex(index).to_numpy(dtype=np.float32)
            audit.append(dict(cell_line=cell,run=path.stem,n_valid_source_sites=len(d),n_matched_observed=int(np.isfinite(matrix[:,j]).sum())))
            del d
        matrices[cell]=matrix
    return matrices,pd.DataFrame(audit)


def run_summary(d, mask, group):
    selected=d.loc[mask]
    row=dict(group=group,n_sites=len(selected),n_strict=int(selected.replicate_supported.sum()),
             strict_pct=100*ratio(int(selected.replicate_supported.sum()),len(selected)),
             n_D_gt_max_range=int(selected.D_gt_max_range.sum()),D_gt_range_pct=100*ratio(int(selected.D_gt_max_range.sum()),len(selected)))
    metrics=["abs_delta_mean","range_Hct116","range_K562","max_within_range","std_score_Hct116","std_score_K562","max_within_sd"]
    metrics += [c for c in ("within_Hct116_mean_pair_abs","within_K562_mean_pair_abs","between_run_mean_pair_abs") if c in d]
    for col in metrics:
        row.update({f"{col}_{k}":v for k,v in describe(selected[col]).items()})
    return row


def ecdf_grid(values):
    values=np.sort(np.asarray(values,dtype=float))
    values=values[np.isfinite(values)]
    grid=np.unique(np.r_[np.linspace(0,1,1001),BASE_DELTA])
    return pd.DataFrame(dict(score_difference=grid,ecdf=np.searchsorted(values,grid,side="right")/len(values) if len(values) else np.full(len(grid),np.nan)))


def analysis_b(tables,destination,raw_root=None):
    matched,original,strict=build_differential_pair(tables,"Hct116","K562",MIN_RUNS,BASE_DELTA)
    d=variability(matched)
    for name,expected in (("selected",original),("strict",strict)):
        saved=pd.read_csv(tables/f"differential_{name}_Hct116_vs_K562.csv")
        if set(map(tuple,saved[KEY].to_numpy()))!=set(map(tuple,expected[KEY].to_numpy())):
            raise ValueError(f"Original {name} baseline site-set regression failed")
    if set(map(tuple,d.loc[d.baseline_selected,KEY].to_numpy()))!=set(map(tuple,original[KEY].to_numpy())):
        raise ValueError("Exact focus baseline differs from canonical atlas tolerance baseline")
    raw_status="unavailable; cached extrema and SDs remain valid; individual run-pair distributions omitted"
    if raw_root is not None:
        matrices,audit=raw_run_matrices(raw_root,tables,d)
        a,b=matrices["Hct116"],matrices["K562"]
        for cell,array in matrices.items():
            count=np.isfinite(array).sum(axis=1)
            if not np.array_equal(count,d[f"n_detected_runs_{cell}"]):
                raise ValueError("Raw detection counts differ from the cached matched universe")
            for stat,calculated in (("mean_score",np.nanmean(array.astype(float),axis=1)),
                                    ("min_score",np.nanmin(array,axis=1)),("max_score",np.nanmax(array,axis=1)),
                                    ("std_score",np.nanstd(array.astype(float),axis=1,ddof=1))):
                if not np.allclose(calculated,d[f"{stat}_{cell}"],atol=1e-7,rtol=1e-5):
                    raise ValueError(f"Raw {stat} differs from cached {cell} summary")
            d[f"within_{cell}_mean_pair_abs"],d[f"n_within_pairs_{cell}"]=mean_pair_differences(array)
        d["between_run_mean_pair_abs"],d["n_cross_run_pairs"]=mean_pair_differences(a,b)
        audit.to_csv(destination/"b_focus_run_coverage.csv",index=False)
        raw_status="complete; six focus run files validated against cached count/mean/min/max/sample SD"
        del matrices,a,b
    groups=[("all_matched",np.ones(len(d),dtype=bool)),("baseline_candidates",d.baseline_selected),
            ("other_matched",~d.baseline_selected),("strict_candidates",d.baseline_selected&d.replicate_supported),
            ("non_strict_candidates",d.baseline_selected&~d.replicate_supported)]
    summaries=pd.DataFrame([run_summary(d,mask,name) for name,mask in groups])
    summaries.to_csv(destination/"b_variability_summary.csv",index=False)
    paired=[]
    for name,mask in groups[:3]:
        sub=d.loc[mask]
        if "between_run_mean_pair_abs" in sub:
            within=np.maximum(sub.within_Hct116_mean_pair_abs,sub.within_K562_mean_pair_abs)
            paired.append(dict(group=name,n_sites=len(sub),
                D_gt_both_within_mean_pair_pct=100*ratio(int(sub.abs_delta_mean.gt(within).sum()),len(sub)),
                between_runs_gt_both_within_mean_pair_pct=100*ratio(int(sub.between_run_mean_pair_abs.gt(within).sum()),len(sub))))
    if paired:
        pd.DataFrame(paired).to_csv(destination/"b_paired_site_comparisons.csv",index=False)
    contingency=d[d.baseline_selected].groupby(["replicate_supported","D_gt_max_range"],as_index=False).size().rename(columns={"size":"n_sites"})
    contingency.to_csv(destination/"b_candidate_consistency_contingency.csv",index=False)
    strata=[]
    for (nh,nk),sub in d.groupby(["n_detected_runs_Hct116","n_detected_runs_K562"],sort=True):
        for label,mask in (("selected",sub.baseline_selected),("other",~sub.baseline_selected)):
            row=run_summary(sub,mask,label)
            row.update(n_observed_Hct116=int(nh),n_observed_K562=int(nk))
            strata.append(row)
    pd.DataFrame(strata).to_csv(destination/"b_coverage_stratified_summary.csv",index=False)
    thresholds=[]
    for t in (.30,.35,.40,.50,.60,.70):
        mask=candidate_mask(d,MAIN_T,t)
        row=run_summary(d,mask,"threshold_selected");row.update(positive_threshold=MAIN_T,delta_threshold=t,n_eligible=len(d))
        thresholds.append(row)
    pd.DataFrame(thresholds).to_csv(destination/"b_difference_thresholds.csv",index=False)
    cover=d.groupby(["n_detected_runs_Hct116","n_detected_runs_K562"],as_index=False).agg(n_sites=("transcript_position","size"),n_baseline_selected=("baseline_selected","sum"))
    cover.to_csv(destination/"b_run_coverage.csv",index=False)
    ecdfs=[]
    for group,mask in groups[:3]:
        for col in ["abs_delta_mean"]+[c for c in ("within_Hct116_mean_pair_abs","within_K562_mean_pair_abs","between_run_mean_pair_abs") if c in d]:
            frame=ecdf_grid(d.loc[mask,col]);frame["metric"]=col;frame["group"]=group;frame["n_sites"]=int(mask.sum());ecdfs.append(frame)
    pd.concat(ecdfs,ignore_index=True).to_csv(destination/"b_ecdf.csv",index=False)
    d[d.baseline_selected].to_csv(destination/"b_baseline_candidate_variability.csv",index=False)
    diagnostics=[]
    for group,mask in groups[:3]:
        subset=d.loc[mask]; cutoff=subset.max_within_range.quantile(.99)
        tail=subset.max_within_range.ge(cutoff)
        global_tail=d.max_within_range.ge(d.max_within_range.quantile(.99))
        diagnostics.append(dict(group=group,n_sites=len(subset),q99_max_within_range=cutoff,n_at_or_above_q99=int(tail.sum()),
            top1pct_share_sum_within_ranges=ratio(subset.loc[tail,"max_within_range"].sum(),subset.max_within_range.sum()),
            pct_in_global_top1pct_variability=100*ratio(int((mask&global_tail).sum()),len(subset)),
            pct_sites_within_range_gt_D=100*ratio(int(subset.max_within_range.gt(subset.abs_delta_mean).sum()),len(subset))))
    pd.DataFrame(diagnostics).to_csv(destination/"b_variability_tail_diagnostics.csv",index=False)
    return d,summaries,dict(n_matched=len(matched),n_eligible=len(d),n_removed=len(matched)-len(d),
                            n_baseline_selected=int(d.baseline_selected.sum()),n_baseline_strict=int((d.baseline_selected&d.replicate_supported).sum()),raw_pair_data=raw_status)


def figure_a(summary, distribution, cells, figures):
    setup_style()
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout="constrained")
    baseline=distribution[(distribution.threshold==MAIN_T)&(distribution.n_positive_cells>0)]
    axes[0].bar(baseline.n_positive_cells,baseline.n_sites,color="#4477AA",width=.72)
    for row in baseline.itertuples():
        axes[0].text(row.n_positive_cells,row.n_sites+70,f"{row.n_sites:,}",ha="center",fontsize=10)
    axes[0].set(xlabel="Cell lines calling a site positive",ylabel="Sites",title="A  Sharing within the seven-cell common universe",xticks=range(1,8),ylim=(0,baseline.n_sites.max()*1.16))
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    baseline=cells[cells.threshold.eq(MAIN_T)].set_index("cell_line").reindex(CELLS)
    palette=get_palette(CELLS)
    axes[1].bar(range(7),baseline.specific_pct_cell_positive,color=[palette[c] for c in CELLS],width=.72)
    for i,row in enumerate(baseline.itertuples()):
        axes[1].text(i,row.specific_pct_cell_positive+.7,f"{row.n_specific:,}",ha="center",fontsize=10)
    axes[1].set(xlabel="Only positive cell line",ylabel="Unique calls / positive calls in that cell (%)",title="B  Cell-line-specific predicted positives",xticks=range(7),xticklabels=CELLS,ylim=(0,baseline.specific_pct_cell_positive.max()*1.2))
    axes[1].tick_params(axis="x",rotation=25)
    row=summary[summary.threshold.eq(MAIN_T)].iloc[0]
    fig.suptitle(f"{int(row.n_eligible):,} eligible sites · positive cutoff 0.5 · ≥2 runs in each cell line",fontsize=12)
    fig.text(.5,-.045,f"A excludes {int(row.n_all_negative):,} sites negative in all seven cells. B labels show counts; each bar uses its cell's comparable positive set.",ha="center",fontsize=10)
    savefig("fig5_site_conservation",figures)


def figure_b(d,ecdf,figures):
    setup_style()
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    labels={"abs_delta_mean":("Consensus-mean difference D","#AA3377","-"),
            "within_Hct116_mean_pair_abs":("Within Hct116: mean pair difference","#4477AA","-"),
            "within_K562_mean_pair_abs":("Within K562: mean pair difference","#EE8866","-"),
            "between_run_mean_pair_abs":("Between individual runs: mean pair difference","#228833","--")}
    for col,(label,color,style) in labels.items():
        rows=ecdf[(ecdf.group=="all_matched")&(ecdf.metric==col)]
        if len(rows):axes[0].plot(rows.score_difference,rows.ecdf,label=label,color=color,linestyle=style,linewidth=1.7)
    axes[0].set(xlabel="Absolute score difference (zoom: 0–0.10)",ylabel="Fraction of sites at or below this difference",title="A  Within- and between-cell score differences",xlim=(0,.10),ylim=(0,1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False,fontsize=8,loc="lower right")
    hb=axes[1].hexbin(d.max_within_range,d.abs_delta_mean,gridsize=65,mincnt=1,bins="log",cmap="Greys",linewidths=0)
    for supported,label,color in ((False,"Selected, overlapping ranges","#EE8866"),(True,"Selected, strict range separation","#AA3377")):
        sub=d[d.baseline_selected&d.replicate_supported.eq(supported)]
        axes[1].scatter(sub.max_within_range,sub.abs_delta_mean,s=5,c=color,alpha=.45,rasterized=True,label=label)
    axes[1].axhline(BASE_DELTA,color="#777777",ls=":",lw=1,label="Baseline difference threshold (0.30)")
    axes[1].plot([0,1],[0,1],color="#777777",ls="--",lw=1,label="D = maximum within-cell range")
    axes[1].set(xlabel="Maximum observed within-cell score range",ylabel="Absolute consensus-mean difference D",title="B  Mean differences versus observed run variation",xlim=(0,1),ylim=(0,1))
    fig.colorbar(hb,ax=axes[1],label="Matched sites (log count)",shrink=.8)
    axes[1].legend(frameon=True,fontsize=8,loc="upper right")
    fig.suptitle(f"Hct116 vs K562 · {len(d):,} sites observed in ≥2 runs per cell",fontsize=12)
    fig.text(.5,-.045,"Each ECDF uses all matched sites (one value/site); A zooms the x-axis only. Selected sites also require higher mean ≥0.5; ranges are not CIs.",ha="center",fontsize=10)
    savefig("fig6_between_vs_within_variability",figures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"outputs")
    parser.add_argument("--input-dir",type=Path,help="External read-only prediction directory; never exported")
    args=parser.parse_args()
    destination=args.output/"tables/extended_analysis";destination.mkdir(parents=True,exist_ok=True)
    tables=args.output/"tables";figures=args.output/"figures"
    raw=args.input_dir or (Path(os.environ["TASK2_RAW_DIR"]) if os.environ.get("TASK2_RAW_DIR") else None)
    summary,distribution,cells,specific=analysis_a(tables,destination)
    print(summary.to_string(index=False),flush=True)
    d,summaries,baseline=analysis_b(tables,destination,raw)
    figure_a(summary,distribution,cells,figures)
    figure_b(d,pd.read_csv(destination/"b_ecdf.csv"),figures)
    metadata=dict(site_key=KEY,cell_lines=list(CELLS),positive_threshold=MAIN_T,min_observed_runs=MIN_RUNS,
        sharing_thresholds=list(SENSITIVITY),sharing_comparison="exact >=, as in cross-cell consensus analysis",
        differential_comparison_tolerance=EPS,baseline=baseline,
        consensus="Mean of available valid run scores; missing runs not zero",std_definition="Sample SD (ddof=1), reused from validated consensus",
        strict_support="Canonical conservative_gap > 0; higher-cell observed minimum exceeds lower-cell observed maximum",
        within_pair_definition="Mean absolute score difference over available unordered run pairs within a site",
        cross_pair_definition="Mean absolute score difference over available Hct116-by-K562 run combinations within a site",
        ecdf="Exact empirical fractions evaluated at fixed 0.001-spaced score differences; plotting connects these grid points; summaries/quantiles are exact",
        interpretation="Descriptive prediction consistency; no independent biological replicate model, hypothesis tests or experimental validation")
    (destination/"methods.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
    print(summaries[["group","n_sites","strict_pct","D_gt_range_pct","abs_delta_mean_median","max_within_range_median"]].to_string(index=False))


if __name__=="__main__":
    main()
