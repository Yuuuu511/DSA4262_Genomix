"""Prediction QC from existing Task 2 outputs; never changes scientific outputs."""
from pathlib import Path
import argparse
import re
import sqlite3
import tempfile
import os

import numpy as np
import pandas as pd

from task2_pipeline import KEY, MAIN_T, MIN_RUNS, build_differential_pair, discover, read_prediction
from run_differential import BASE_DELTA, EPS, POS_THRESHOLDS, candidate_mask, transcript_summary

BASE_POSITIVE = MAIN_T
LOCAL_POSITIVES = (.40, .45, .50, .55, .60)
LOCAL_DELTAS = (.20, .25, .30, .35, .40)
BROAD_DELTAS = (.3, .4, .5, .6, .7)
QUANTILES = {"score_q01": .01, "score_q05": .05, "score_q25": .25,
             "score_q75": .75, "score_q95": .95, "score_q99": .99}
VALIDATION = ["missing_transcript_ids", "missing_positions", "invalid_positions",
              "missing_scores", "invalid_scores", "duplicate_site_records",
              "nonfinite_scores", "out_of_range_scores"]
RANKING = "strict site count desc; selected site count desc; max absolute delta desc; transcript ID asc (tie break)"


def safe_rate(n, d):
    return n / d if d else np.nan


def raw_run_qc(path, chunksize=100_000):
    """Exact chunked audit: disk-backed canonical keys, one compact score vector.

    SQLite upserts count duplicate keys across all chunks. No row is chosen from
    a duplicate group. Exact quantiles/moments use at most one run's float32 score
    vector (4 bytes per valid site), in original row order, matching the pipeline.
    Temporary databases are outside the repository and removed on completion.
    """
    counts = {k: 0 for k in VALIDATION}
    counts.update(n_records=0, nan_scores=0, infinite_scores=0)
    with tempfile.TemporaryDirectory(prefix="task2-qc-") as tmp:
        with sqlite3.connect(str(Path(tmp) / "sites.sqlite")) as db:
            db.execute("PRAGMA cache_size = -16384")
            db.execute("PRAGMA temp_store = FILE")
            db.execute("CREATE TABLE sites (id TEXT, pos INTEGER, n INTEGER, score REAL, valid INTEGER, PRIMARY KEY(id,pos))")
            for d in pd.read_csv(path, usecols=KEY + ["score"], dtype="string", chunksize=chunksize):
                ids = d.transcript_id.mask(d.transcript_id.str.strip().eq(""))
                ids = ids.str.replace(r"^(ENST\d+)\.\d+$", r"\1", regex=True)
                pos = pd.to_numeric(d.transcript_position, errors="coerce")
                scores = pd.to_numeric(d.score, errors="coerce").astype("float32")
                missing_pos = d.transcript_position.isna() | d.transcript_position.str.strip().eq("").fillna(False)
                missing_score = d.score.isna() | d.score.str.strip().eq("").fillna(False)
                valid_pos = pos.notna() & np.isfinite(pos) & pos.eq(np.floor(pos)) & pos.between(-2147483648, 2147483647)
                valid_score = np.isfinite(scores) & scores.between(0, 1)
                counts["n_records"] += len(d)
                for key, mask in {
                    "missing_transcript_ids": ids.isna(), "missing_positions": missing_pos,
                    "invalid_positions": ~missing_pos & ~valid_pos,
                    "missing_scores": missing_score, "invalid_scores": ~missing_score & scores.isna(),
                    "nonfinite_scores": ~np.isfinite(scores), "nan_scores": scores.isna(),
                    "infinite_scores": np.isinf(scores),
                    "out_of_range_scores": np.isfinite(scores) & ~scores.between(0, 1),
                }.items():
                    counts[key] += int(mask.sum())
                keys = pd.DataFrame({"id": ids, "pos": pos, "score": scores, "valid": valid_score})[ids.notna() & valid_pos]
                db.executemany(
                    "INSERT INTO sites VALUES (?,?,1,?,?) ON CONFLICT(id,pos) DO UPDATE SET n=n+1",
                    ((str(i), int(p), float(v) if np.isfinite(v) else None, int(ok))
                     for i, p, v, ok in keys.itertuples(index=False, name=None)),
                )
                db.commit()
            unique, duplicates, members = db.execute(
                "SELECT count(*),coalesce(sum(n-1),0),coalesce(sum(CASE WHEN n>1 THEN n ELSE 0 END),0) FROM sites"
            ).fetchone()
            nvalid = db.execute("SELECT count(*) FROM sites WHERE n=1 AND valid=1").fetchone()[0]
            values = np.fromiter((r[0] for r in db.execute("SELECT score FROM sites WHERE n=1 AND valid=1 ORDER BY rowid")), dtype=np.float32, count=nvalid)
    counts["duplicate_site_records"] = duplicates
    finite = pd.Series(values)
    npositive = int(finite.ge(MAIN_T).sum())
    issues = [f"{k}={counts[k]}" for k in VALIDATION if counts[k]]
    if not counts["n_records"]:
        issues.append("empty prediction file")
    result = dict(counts, n_unique_sites=unique, duplicate_member_records=members,
                  n_valid_records=nvalid, n_invalid_records=counts["n_records"]-nvalid,
                  valid_record_pct=100*safe_rate(nvalid, counts["n_records"]),
                  n_valid_scored_sites=nvalid, n_predicted_sites=nvalid,
                  n_positive=npositive, positive_fraction=safe_rate(npositive,nvalid),
                  positive_rate_pct=100*safe_rate(npositive,nvalid), mean_score=finite.mean(),
                  median_score=finite.median(), score_sd=finite.std(),
                  metric_source="raw_csv", validation_status="checked", structural_validation_issue=False,
                  source_path=f"{path.parent.name}/{path.name}",
                  aggregation_method="Exact SQLite canonical-key counts across chunks; exact one-run float32 score vector",
                  n_pipeline_retained_records=0 if any(counts[k] for k in VALIDATION) else counts["n_records"],
                  pipeline_input_handling="Existing loader rejects entire file on validation issues" if issues else "Meets existing loader constraints; no records filtered",
                  input_warnings="; ".join(issues))
    result.update({k: finite.quantile(q) for k,q in QUANTILES.items()})
    return result


def verify_raw_roster(tables, raw_root):
    """Check all source identities against the analyzed roster before writing QC."""
    if not raw_root.is_dir():
        raise ValueError("External raw-input directory is unavailable")
    repo = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), Path(__file__).resolve().parent)
    if raw_root.resolve().is_relative_to(repo):
        raise ValueError("Raw predictions must remain outside the Git repository")
    files = discover(raw_root)
    roster = {(cell,p.stem) for cell,paths in files.items() for p in paths}
    expected_path = tables / "prediction_overview.csv"
    if expected_path.exists():
        expected = pd.read_csv(expected_path)
        expected_roster = set(map(tuple,expected[["cell_line","run"]].to_numpy()))
        if roster != expected_roster or sum(map(len,files.values())) != len(expected):
            raise ValueError(f"Raw roster mismatch: {len(roster)} sources vs {len(expected)} analyzed runs")
    print(f"Verified external raw roster: {len(roster)} runs across {len(files)} cell lines", flush=True)
    return files


def input_qc(tables, raw_root=None, chunksize=100_000):
    overview_path = tables / "prediction_overview.csv"
    old = pd.read_csv(overview_path) if overview_path.exists() else pd.DataFrame(columns=["cell_line", "run"])
    paths = {}
    if raw_root is not None and raw_root.exists():
        try:
            paths = {p.stem: (cell, p) for cell, ps in discover(raw_root).items() for p in ps}
        except ValueError:
            pass
    known = {r["run"]: r for r in old.to_dict("records")}
    rows = []
    for run in sorted(set(known) | set(paths)):
        prior = known.get(run, {})
        row = {"cell_line": prior.get("cell_line", paths.get(run, (None,))[0]),
               "run": run, "baseline_positive_threshold": MAIN_T,
               "structural_validation_issue": np.nan}
        if run in paths:
            try:
                print(f"Auditing {run}", flush=True)
                row.update(raw_run_qc(paths[run][1], chunksize))
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                row.update(metric_source="unreadable_raw_csv", validation_status="unavailable",
                           structural_validation_issue=True if not isinstance(exc, OSError) else np.nan,
                           input_warnings=f"Raw CSV could not be audited ({type(exc).__name__}); check required columns, CSV syntax and access")
        if row.get("validation_status") != "checked":
            # Cached counts describe accepted records, not a fresh inspection of raw inputs.
            row.update({k: np.nan for k in VALIDATION + list(QUANTILES)})
            n = prior.get("n_sites", np.nan)
            positive = prior.get(f"positive_{MAIN_T}", np.nan)
            row.update(n_records=np.nan, n_valid_records=np.nan, n_invalid_records=np.nan,
                       valid_record_pct=np.nan, duplicate_member_records=np.nan,
                       n_pipeline_retained_records=np.nan, n_unique_sites=n,
                       n_predicted_sites=n, n_valid_scored_sites=n, n_positive=positive,
                       positive_rate_pct=100 * safe_rate(positive, n),
                       pipeline_input_handling="Current raw input unavailable; cached counts describe historical accepted records",
                       positive_fraction=safe_rate(positive, n),
                       **{k: prior.get(k, np.nan) for k in ("mean_score", "median_score", "score_sd")})
            row.setdefault("metric_source", "cached_prediction_overview")
            row.setdefault("validation_status", "unavailable")
            row.setdefault("source_path", "prediction_overview.csv")
            row.setdefault("input_warnings", "Raw CSV unavailable: validation counts and score quantiles cannot be reconstructed")
        row["n_cached_retained_records"] = prior.get("n_sites", np.nan)
        row["analyzed_run"] = run in known if known else True
        row["source_correspondence"] = "Raw unavailable; cached run identity only"
        if row["validation_status"] == "checked":
            if prior:
                match = row["n_pipeline_retained_records"] == prior["n_sites"] and row["n_positive"] == prior[f"positive_{MAIN_T}"]
                for field in ("mean_score", "median_score", "score_sd"):
                    row[f"cached_{field}_delta"] = row[field] - prior[field]
                    match = match and bool(np.isclose(row[field], prior[field], rtol=1e-5, atol=1e-7, equal_nan=True))
                row["cached_statistics_agree"] = match
                row["source_correspondence"] = "Run identity, site/positive counts and score moments agree with cache" if match else "Raw metrics differ from analyzed cache; scientific outputs unchanged"
                if not match:
                    row["input_warnings"] += "; Raw metrics differ from analyzed cache"
            else:
                row["source_correspondence"] = "No cached analyzed run to verify"
        row["has_input_warning"] = bool(row["input_warnings"])
        rows.append(row)
    return pd.DataFrame(rows, columns=None if rows else ["cell_line", "run", "validation_status", "has_input_warning"])


def cached_input_qc(tables):
    """Cached overview fallback only; no raw discovery, environment lookup or CSV audit."""
    return input_qc(tables, raw_root=None)


def read_optional(path, columns):
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def reproducibility_qc(tables, runs):
    pairs = read_optional(tables / "replicate_pairwise.csv", ["cell_line", "run_a", "run_b"])
    coverage = read_optional(tables / "replicate_detection_coverage.csv", ["cell_line"])
    rows = []
    for cell, group in runs.groupby("cell_line", sort=True):
        cp = pairs[pairs.cell_line == cell]
        cov = coverage[coverage.cell_line == cell]
        n = len(group)
        for run in sorted(group.run):
            p = cp[(cp.run_a == run) | (cp.run_b == run)]
            overlap = pd.concat([p.loc[p.run_a == run, "fraction_a"], p.loc[p.run_b == run, "fraction_b"]]) if len(p) else pd.Series(dtype=float)
            same_rep = sum(re.search(r"_rep(\d+)_run", a).group(1) == re.search(r"_rep(\d+)_run", b).group(1)
                           for a, b in zip(p.run_a, p.run_b))
            rows.append({
                "cell_line": cell, "run": run, "n_cell_runs": n,
                "n_partner_runs": len(p), "same_replicate_label_pairs": same_rep,
                "different_replicate_label_pairs": len(p) - same_rep,
                "mean_pearson": p.pearson.mean() if len(p) else np.nan,
                "min_pearson": p.pearson.min() if len(p) else np.nan,
                "max_pearson": p.pearson.max() if len(p) else np.nan,
                "mean_spearman": p.spearman.mean() if len(p) else np.nan,
                "mean_shared_positive_jaccard_0.5": p["jaccard_0.5"].mean() if len(p) else np.nan,
                "mean_fraction_of_run_sites_shared": overlap.mean(),
                "min_fraction_of_run_sites_shared": overlap.min(),
                "mean_shared_sites": p.n_shared.mean() if len(p) else np.nan,
                "cell_union_sites": cov.n_sites.sum() if len(cov) else np.nan,
                "cell_fraction_sites_detected_all_runs": safe_rate(cov.loc[cov.n_detected_runs == n, "n_sites"].sum(), cov.n_sites.sum()) if len(cov) else np.nan,
                "cell_fraction_sites_detected_at_least_two_runs": safe_rate(cov.loc[cov.n_detected_runs >= 2, "n_sites"].sum(), cov.n_sites.sum()) if len(cov) else np.nan,
                "metadata_limit": "Replicate IDs inferred from filenames; biological independence unverified; repeated labels may be technical runs",
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["cell_line", "run", "warnings"])
    for cell, idx in result.groupby("cell_line").groups.items():
        vals = result.loc[idx, "mean_pearson"]
        q1, q3 = vals.quantile([.25, .75])
        fence = q1 - 1.5 * (q3 - q1)
        for i in idx:
            row = result.loc[i]
            notes = []
            if row.n_partner_runs == 0:
                notes.append("Pairwise agreement unavailable")
            elif row.n_cell_runs == 2:
                notes.append("Only two runs: one pair; cannot attribute low agreement to either run or estimate between-pair variability")
            else:
                if row.mean_pearson == vals.min() and vals.max() > vals.min():
                    notes.append("Lowest mean shared-site Pearson in this cell (descriptive rank only)")
                if row.mean_pearson < fence:
                    notes.append("Unusually low mean Pearson: below cell run-mean Q1 - 1.5 IQR (descriptive warning, not exclusion)")
            if row.same_replicate_label_pairs:
                notes.append("Repeated replicate label: pair comparisons may include technical runs")
            if pd.isna(row.cell_union_sites):
                notes.append("Detection coverage table unavailable")
            result.loc[i, "cell_median_run_mean_pearson"] = vals.median()
            result.loc[i, "cell_pearson_lower_tukey_fence"] = fence
            result.loc[i, "warnings"] = "; ".join(notes)
    return result


def assess_runs(inputs, rep):
    """Descriptive within-cell Tukey fences; metadata limitations are context, not failures."""
    result = inputs.copy()
    if result.empty:
        return result.assign(descriptive_warnings="", has_descriptive_warning=False,
                             complete_input_qc=False, qc_status="Input QC unavailable")
    columns = ["cell_line", "run", "mean_pearson", "mean_spearman",
               "mean_shared_positive_jaccard_0.5", "mean_fraction_of_run_sites_shared", "warnings"]
    if not rep.empty:
        result = result.merge(rep.reindex(columns=columns), on=["cell_line", "run"], how="left", validate="one_to_one")
    else:
        for col in columns[2:]:
            result[col] = np.nan
    required = VALIDATION + ["n_records", "n_valid_records", "valid_record_pct"]
    checked = result.get("validation_status", pd.Series("unavailable", index=result.index)).eq("checked")
    result["complete_input_qc"] = checked & result.reindex(columns=required).notna().all(axis=1)
    result["input_validation_coverage"] = np.where(result.complete_input_qc, "Complete raw audit", "Raw audit unavailable or incomplete")
    result["descriptive_warnings"] = ""
    result["qc_context"] = result["warnings"].fillna("")
    rules = [("n_predicted_sites", "site count", "both", True),
             ("positive_rate_pct", "positive rate at 0.5", "both", False),
             ("mean_score", "mean score", "both", False),
             ("mean_pearson", "shared-site Pearson", "low", False),
             ("mean_fraction_of_run_sites_shared", "site overlap", "low", False)]
    for cell, idx in result.groupby("cell_line", sort=True).groups.items():
        for field, label, direction, logarithm in rules:
            values = pd.to_numeric(result.loc[idx, field], errors="coerce")
            if logarithm:
                values = np.log2(values.where(values > 0))
            available = values.dropna()
            if len(available) < 3:
                continue  # Two runs cannot support attribution to one run.
            q1, q3 = available.quantile([.25, .75])
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            result.loc[idx, field + "_lower_fence"] = low
            result.loc[idx, field + "_upper_fence"] = high
            for i, value in values.items():
                if value < low or (direction == "both" and value > high):
                    side = "below Q1 - 1.5 IQR" if value < low else "above Q3 + 1.5 IQR"
                    reason = f"{label}: {side} within {cell}" + (" (log2 counts)" if logarithm else "")
                    result.loc[i, "descriptive_warnings"] += reason + "; "
        if len(idx) == 2:
            counts = result.loc[idx, "n_predicted_sites"]
            rates = result.loc[idx, "positive_rate_pct"]
            context = f"Two runs: site-count ratio {safe_rate(counts.max(), counts.min()):.2f}; positive-rate span {rates.max() - rates.min():.3f} percentage points. Outlier attribution ambiguous."
            result.loc[idx, "qc_context"] += "; " + context
    result["descriptive_warnings"] = result.descriptive_warnings.str.rstrip("; ")
    result["has_descriptive_warning"] = result.descriptive_warnings.ne("")
    issues = result.get("input_warnings", pd.Series("", index=result.index)).fillna("")
    observed_problem = result.reindex(columns=VALIDATION).apply(pd.to_numeric, errors="coerce").gt(0).any(axis=1) | (result.complete_input_qc & issues.ne(""))
    observed_problem |= result.get("structural_validation_issue", pd.Series(np.nan, index=result.index)).eq(True)
    result["has_observed_input_problem"] = observed_problem
    result["qc_status"] = np.select(
        [observed_problem | result.has_descriptive_warning, ~result.complete_input_qc],
        ["Review recommended", "Input QC unavailable"], default="No issues detected")
    return result


def dashboard_run_qc(tables):
    """Join optional/partial QC onto the cached run roster without inventing audit data."""
    inputs = cached_input_qc(tables)
    path = tables / "qc_run_summary.csv"
    if path.exists():
        try:
            saved = pd.read_csv(path)
            ready = {"cell_line", "run", "complete_input_qc", "has_descriptive_warning",
                     "qc_status", "input_validation_coverage", "descriptive_warnings", "qc_context",
                     "n_predicted_sites", "n_positive", "positive_rate_pct", "mean_score", "metric_source",
                     "mean_pearson", "mean_spearman", "mean_shared_positive_jaccard_0.5",
                     "mean_fraction_of_run_sites_shared"} | set(VALIDATION)
            if ready.issubset(saved.columns) and len(saved) == len(inputs):
                roster = set(map(tuple, inputs[["cell_line","run"]].to_numpy()))
                if (not saved.duplicated(["cell_line","run"]).any()
                        and set(map(tuple,saved[["cell_line","run"]].to_numpy())) == roster
                        and saved.complete_input_qc.eq(True).all()
                        and saved[list(VALIDATION)].notna().all().all()):
                    return saved  # Deployment: fully precomputed artifacts, no audit/recalculation.
            if {"run", "cell_line"}.issubset(saved.columns):
                # Roster from the scientific overview remains authoritative.
                inputs = inputs.set_index(["cell_line", "run"])
                saved = saved.drop_duplicates(["cell_line", "run"]).set_index(["cell_line", "run"])
                population = ["n_positive", "n_valid_scored_sites", "n_predicted_sites"]
                consistent_population = saved.reindex(columns=population).notna().all(axis=1)
                for col in saved.columns:
                    values = saved[col]
                    if col in population + ["positive_fraction", "positive_rate_pct"]:
                        values = values.where(consistent_population)
                    inputs[col] = values.reindex(inputs.index).combine_first(inputs[col]) if col in inputs else values.reindex(inputs.index)
                inputs = inputs.reset_index()
        except (ValueError, OSError, pd.errors.ParserError):
            pass
    if inputs.empty:
        return assess_runs(inputs, pd.DataFrame())
    inputs["positive_fraction"] = inputs.n_positive / inputs.n_valid_scored_sites.where(inputs.n_valid_scored_sites > 0)
    inputs["positive_rate_pct"] = 100 * inputs.positive_fraction
    rep = reproducibility_qc(tables, inputs)
    return assess_runs(inputs.drop(columns=[c for c in ("warnings", "mean_pearson", "mean_spearman", "mean_shared_positive_jaccard_0.5", "mean_fraction_of_run_sites_shared") if c in inputs]), rep)


def heya8_shared_qc(path_a, path_b, tables):
    """Read only the validated HEYA8 pair, never all 23 runs together."""
    a = read_prediction(path_a)
    b = read_prediction(path_b)
    matched = a.merge(b, on=KEY, suffixes=("_a", "_b"), how="inner", validate="one_to_one")
    rows = []
    for suffix, raw, path in (("a",a,path_a),("b",b,path_b)):
        shared = matched[f"score_{suffix}"]
        unshared = raw.loc[~pd.MultiIndex.from_frame(raw[KEY]).isin(pd.MultiIndex.from_frame(matched[KEY])), "score"]
        for population, scores in (("all_sites",raw.score),("shared_sites",shared),("run_only_sites",unshared)):
            row = dict(run=path.stem, population=population, n_sites=len(scores),
                       mean_score=scores.mean(), median_score=scores.median(), score_sd=scores.std(),
                       n_positive=int(scores.ge(MAIN_T).sum()), positive_rate_pct=100*safe_rate(int(scores.ge(MAIN_T).sum()),len(scores)))
            row.update({k:scores.quantile(q) for k,q in QUANTILES.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(tables / "qc_heya8_score_distributions.csv", index=False)
    x,y = matched.score_a, matched.score_b
    pair = dict(run_a=path_a.stem, run_b=path_b.stem, n_a=len(a), n_b=len(b), n_shared=len(matched),
                fraction_a=safe_rate(len(matched),len(a)), fraction_b=safe_rate(len(matched),len(b)),
                site_jaccard=safe_rate(len(matched),len(a)+len(b)-len(matched)),
                n_a_only=len(a)-len(matched), n_b_only=len(b)-len(matched),
                pearson=x.corr(y), spearman=x.corr(y,method="spearman"),
                mean_paired_score_difference=(x-y).mean(), median_paired_score_difference=(x-y).median(),
                positive_rate_shared_a_pct=100*x.ge(MAIN_T).mean(), positive_rate_shared_b_pct=100*y.ge(MAIN_T).mean(),
                shared_positive_both=int((x.ge(MAIN_T)&y.ge(MAIN_T)).sum()),
                shared_positive_jaccard=safe_rate(int((x.ge(MAIN_T)&y.ge(MAIN_T)).sum()),int((x.ge(MAIN_T)|y.ge(MAIN_T)).sum())),
                positive_a_only=int((x.ge(MAIN_T)&~y.ge(MAIN_T)).sum()),
                positive_b_only=int((~x.ge(MAIN_T)&y.ge(MAIN_T)).sum()),
                interpretation="Shared-site shifts persist after controlling site identity; coverage and score differences coexist; neither establishes technical failure")
    cached_path = tables / "replicate_pairwise.csv"
    if cached_path.exists():
        cached = pd.read_csv(cached_path)
        cached = cached[(cached.run_a == path_a.stem) & (cached.run_b == path_b.stem)]
        if len(cached):
            r = cached.iloc[0]
            pair["cached_pairwise_agrees"] = int(r.n_shared)==len(matched) and np.isclose(r.pearson,pair["pearson"]) and np.isclose(r.spearman,pair["spearman"])
    pd.DataFrame([pair]).to_csv(tables / "qc_heya8_shared_sites.csv", index=False)
    return pair


def run_level_figure(summary, folder):
    """Positive rates from actual valid populations; boxes only from measured quantiles."""
    import matplotlib.pyplot as plt
    from task2_pipeline import setup_style, savefig, get_palette
    if summary.empty or not summary.positive_rate_pct.notna().any():
        return
    setup_style()
    d = summary.sort_values(["cell_line", "run"]).reset_index(drop=True)
    quantiles = ["score_q05", "score_q25", "median_score", "score_q75", "score_q95"]
    available = d.reindex(columns=quantiles).notna().all(axis=1)
    n_panels = 2 if available.any() else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11.5, 3.8 * n_panels), squeeze=False, constrained_layout=True)
    palette = get_palette(sorted(d.cell_line.unique()))
    x = np.arange(len(d))
    ax = axes[0, 0]
    ax.bar(x, d.positive_rate_pct, color=[palette[c] for c in d.cell_line], width=.75)
    ax.set(ylabel="Positive prediction rate (%)", ylim=(0, max(.1,d.positive_rate_pct.max()*1.15)))
    ax.set_title("A  Positive prediction rates (fixed score ≥ 0.5)", loc="left", pad=30)
    if n_panels == 2:
        boxes = [dict(med=r.median_score, q1=r.score_q25, q3=r.score_q75,
                      whislo=r.score_q05, whishi=r.score_q95) for r in d[available].itertuples()]
        artists = axes[1,0].bxp(boxes, positions=x[available], showfliers=False, patch_artist=True, widths=.65,
                              medianprops={"color":"#222222","linewidth":1.2})
        for patch,cell in zip(artists["boxes"],d.loc[available,"cell_line"]):
            patch.set_facecolor(palette[cell]); patch.set_alpha(.65)
        axes[1,0].set(ylabel="Prediction score")
        axes[1,0].set_title(f"B  Measured score distributions ({int(available.sum())}/{len(d)} runs)", loc="left", pad=30)
        fig.supxlabel("Boxes: 25th–75th percentiles; line: median; whiskers: 5th–95th percentiles; outliers omitted", fontsize=10)
    else:
        fig.supxlabel("Cached accepted-site summaries; raw score distributions unavailable — distribution panel omitted", fontsize=10)
    for ax in axes[:,0]:
        for cell,g in d.groupby("cell_line",sort=False):
            ax.text((g.index.min()+g.index.max())/2,1.02,cell,transform=ax.get_xaxis_transform(),ha="center",fontsize=10)
            if g.index.min():
                ax.axvline(g.index.min()-.5,color="#dddddd",linewidth=.7)
        ax.set_xticks(x,[r.split("_",1)[1] for r in d.run],rotation=60,ha="right",fontsize=9)
        ax.set_xlim(-.6,len(d)-.4)
        ax.spines[["top","right"]].set_visible(False)
    savefig("figS3_run_level_qc", folder)


def top_transcripts(selected, k=20):
    tx = transcript_summary(selected)
    # Extend the existing ranking only with a deterministic final tie break.
    return tx.sort_values(["n_replicate_supported", "n_diff_sites", "max_abs_delta", "transcript_id"],
                          ascending=[False, False, False, True]).head(k).transcript_id.tolist()


def robustness(d, positives=LOCAL_POSITIVES, deltas=LOCAL_DELTAS):
    """Fixed-universe sensitivity; fractions with empty denominators are NA."""
    if d.duplicated(KEY).any():
        raise ValueError("Matched universe contains duplicate canonical site keys")
    baseline = candidate_mask(d, BASE_POSITIVE, BASE_DELTA)
    baseline_top = top_transcripts(d[baseline])
    rows = []
    for pos in sorted(set(positives) | {BASE_POSITIVE}):
        for delta in sorted(set(deltas) | {BASE_DELTA}):
            mask = candidate_mask(d, pos, delta)
            selected = d[mask]
            n = len(selected)
            overlap = int((mask & baseline).sum())
            union = int((mask | baseline).sum())
            top = top_transcripts(selected)
            common = set(top) & set(baseline_top)
            rows.append({
                "positive_threshold": pos, "delta_threshold": delta,
                "baseline_positive_threshold": BASE_POSITIVE, "baseline_delta_threshold": BASE_DELTA,
                "comparison_tolerance": EPS, "min_detected_runs_per_cell": MIN_RUNS,
                "is_baseline": pos == BASE_POSITIVE and delta == BASE_DELTA,
                "n_matched_sites": len(d), "n_selected": n,
                "n_strict": int(selected.replicate_supported.sum()),
                "strict_fraction": safe_rate(selected.replicate_supported.sum(), n),
                "strict_support_pct": 100 * safe_rate(selected.replicate_supported.sum(), n),
                "n_transcripts": selected.transcript_id.nunique(),
                "n_baseline_selected": int(baseline.sum()), "n_baseline_overlap": overlap,
                "candidate_jaccard_vs_baseline": safe_rate(overlap, union),
                "baseline_retention": safe_rate(overlap, int(baseline.sum())),
                "baseline_retention_pct": 100 * safe_rate(overlap, int(baseline.sum())),
                "top_k": 20, "n_top_transcripts": len(top),
                "top_transcript_jaccard_vs_baseline": safe_rate(len(common), len(set(top) | set(baseline_top))),
                "baseline_top_transcript_retention_pct": 100 * safe_rate(len(common), len(baseline_top)),
                "top_common_mean_absolute_rank_shift": np.mean([abs(top.index(t) - baseline_top.index(t)) for t in sorted(common)]) if common else np.nan,
                "top_transcripts": ";".join(top), "transcript_ranking_rule": RANKING,
            })
    return pd.DataFrame(rows)


def robustness_grids(d):
    """Retain local and historical broad grids without duplicate combinations.

    Semicolon-delimited grid_id records membership in both grids at shared cells.
    """
    local = robustness(d).assign(grid_id="local")
    broad = robustness(d, POS_THRESHOLDS, BROAD_DELTAS).assign(grid_id="broad")
    keys = ["positive_threshold", "delta_threshold"]
    combined = pd.concat([local, broad], ignore_index=True)
    membership = combined.groupby(keys, sort=False).grid_id.agg(";".join)
    return combined.drop(columns="grid_id").drop_duplicates(keys).merge(
        membership, on=keys, how="left", validate="one_to_one"
    )


def robustness_figure(summary, folder):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter
    from task2_pipeline import setup_style, savefig
    setup_style()
    local = summary[summary.grid_id.str.contains("local", regex=False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for ax, col, title, fmt, label in zip(
        axes, ("n_selected", "candidate_jaccard_vs_baseline", "strict_fraction"),
        ("A  Differential candidate counts", "B  Jaccard vs baseline", "C  Strict-support fraction"),
        (",.0f", ".3f", ".1%"),
        ("Selected sites (count)", "Jaccard (0–1)", "Strict-supported sites (%)"),
    ):
        mat = local.pivot(index="positive_threshold", columns="delta_threshold", values=col)
        vmax = max(float(mat.max().max()), 1) if col == "n_selected" else 1
        sns.heatmap(mat, ax=ax, cmap="Blues", vmin=0, vmax=vmax, annot=True,
                    fmt=fmt, annot_kws={"fontsize": 10}, linewidths=.5,
                    cbar_kws={"label": label, "shrink": .83, "pad": .025})
        # Pick black/white annotation text using the rendered cell luminance.
        mesh = ax.collections[0]
        for text in ax.texts:
            x, y = text.get_position()
            value = mat.iloc[int(y), int(x)]
            rgb = np.array(mesh.cmap(mesh.norm(value))[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = linear @ np.array([.2126, .7152, .0722])
            text.set_color("black" if luminance > .179 else "white")
        bar = mesh.colorbar
        bar.formatter = PercentFormatter(1, decimals=0) if col == "strict_fraction" else StrMethodFormatter("{x:,.0f}" if col == "n_selected" else "{x:.1f}")
        bar.update_ticks()
        ax.add_patch(Rectangle((list(mat.columns).index(BASE_DELTA), list(mat.index).index(BASE_POSITIVE)),
                               1, 1, fill=False, edgecolor="#AA3377", linewidth=1.8))
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set(xlabel="", ylabel="")
        ax.set_xticklabels([f"{v:.2f}" for v in mat.columns], rotation=0)
        ax.set_yticklabels([f"{v:.2f}" for v in mat.index], rotation=0)
    fig.supxlabel("Absolute mean-score difference threshold", fontsize=12)
    fig.supylabel("Higher-cell mean-score threshold", fontsize=12)
    fig.suptitle(f"Hct116 vs K562 · fixed matched-site universe (n = {int(local.n_matched_sites.iloc[0]):,})", fontsize=12)
    fig.text(.5, -.035, "Magenta outline: baseline (positive ≥ 0.50, |Δ| ≥ 0.30) · Strict support is descriptive range separation", ha="center", fontsize=10)
    savefig("figS2_qc_robustness", folder)


def run(output, raw_root=None, run_level_only=False, chunksize=100_000):
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if raw_root is not None:
        verify_raw_roster(tables, raw_root)
    inputs = input_qc(tables, raw_root, chunksize)
    rep = reproducibility_qc(tables, inputs)
    inputs = assess_runs(inputs, rep)
    inputs.to_csv(tables / "qc_run_summary.csv", index=False)
    if raw_root is not None:
        paths = {p.stem:p for ps in discover(raw_root).values() for p in ps}
        pair_names = ("HEYA8_rep1_run1", "HEYA8_rep1_run2")
        usable = inputs[inputs.run.isin(pair_names)]
        if (all(n in paths for n in pair_names) and len(usable) == 2
                and usable.complete_input_qc.all() and not usable.has_observed_input_problem.any()):
            heya8_shared_qc(paths[pair_names[0]], paths[pair_names[1]], tables)
    run_level_figure(inputs, output / "figures")
    rep.to_csv(tables / "qc_reproducibility_summary.csv", index=False)
    overview = []
    def record(metric, value, note=""):
        overview.append({"metric": metric, "value": value, "note": note})
    record("total_analyzed_runs", len(inputs))
    record("raw_runs_checked", int(inputs.validation_status.eq("checked").sum()))
    record("runs_with_complete_input_qc", int(inputs.complete_input_qc.sum()))
    for metric in ("n_records", "n_valid_records", "n_invalid_records", "nan_scores", "infinite_scores"):
        values = inputs[metric] if metric in inputs else pd.Series(dtype=float)
        record("total_" + metric, values.sum(min_count=1), "Measured runs only; unavailable values are not zero")
    record("raw_runs_agreeing_with_cache", int(inputs.get("cached_statistics_agree", pd.Series(dtype=bool)).fillna(False).sum()), "Counts exact; score moments rtol=1e-5, atol=1e-7 for float32 arithmetic")
    for metric in VALIDATION:
        values = inputs[metric] if metric in inputs else pd.Series(dtype=float)
        record("total_" + metric, values.sum(min_count=1), "Sum of measured runs only; NA when none audited")
    record("runs_with_descriptive_warnings", int(inputs.has_descriptive_warning.sum()))
    record("runs_with_input_warnings", int(inputs.has_input_warning.sum()), "Includes unavailable raw input audit; not evidence of bad records")
    record("runs_with_observed_input_problems", int(inputs.has_observed_input_problem.sum()) if len(inputs) else 0)
    record("input_validation", "complete" if len(inputs) and inputs.validation_status.eq("checked").all() else "incomplete", "Cached accepted record/site counts and moments are historical; missing validation metrics are NA, never zero")
    record("reproducibility_cell_mean_pearson_min", rep.groupby("cell_line").mean_pearson.mean().min() if len(rep) else np.nan)
    record("reproducibility_cell_mean_pearson_max", rep.groupby("cell_line").mean_pearson.mean().max() if len(rep) else np.nan)
    record("runs_with_reproducibility_warnings", int(rep.warnings.fillna("").ne("").sum()) if len(rep) else 0)
    warned = set(inputs.loc[inputs.has_input_warning | inputs.has_descriptive_warning, "run"])
    record("runs_with_any_qc_warning", len(warned), "Observed input/descriptive anomalies or unavailable audits; metadata dependence is context, not an anomaly")
    sharing = read_optional(tables / "site_sharing.csv", ["n_sites", "n_cell_lines_observed"])
    record("cross_cell_evaluable_union_sites", sharing.n_sites.sum() if len(sharing) else np.nan, "Each cell observation requires ≥2 runs; detection is not modification specificity")
    record("cross_cell_sites_observed_all_cells", sharing.loc[sharing.n_cell_lines_observed == inputs.cell_line.nunique(), "n_sites"].sum() if len(sharing) else np.nan)
    if run_level_only:
        previous = read_optional(tables / "qc_overview.csv", ["metric", "value", "note"])
        updated = pd.DataFrame(overview)
        pd.concat([updated, previous[~previous.metric.isin(updated.metric)]], ignore_index=True).to_csv(tables / "qc_overview.csv", index=False)
        print(updated.to_string(index=False))
        return
    if all((tables / f"consensus_{c}.parquet").exists() for c in ("Hct116", "K562")):
        d, exact, strict = build_differential_pair(tables, "Hct116", "K562", MIN_RUNS, BASE_DELTA)
        summary = robustness_grids(d)
        # Verify both historical focus CSVs and the authoritative atlas against canonical keys.
        for name, expected in (("selected", exact), ("strict", strict)):
            path = tables / f"differential_{name}_Hct116_vs_K562.csv"
            if path.exists():
                saved = pd.read_csv(path)
                same = set(map(tuple, saved[KEY].to_numpy())) == set(map(tuple, expected[KEY].to_numpy())) and len(saved) == len(expected)
                record(f"baseline_{name}_regression_matches", same, "Exact original pipeline comparisons; existing files never written")
                if not same:
                    raise ValueError(f"Baseline {name} regression mismatch")
        baseline = d[candidate_mask(d, BASE_POSITIVE, BASE_DELTA)]
        atlas_path = tables / "differential_atlas" / f"positive_{BASE_POSITIVE:.1f}" / "differential_site_atlas.parquet"
        if atlas_path.exists():
            atlas = pd.read_parquet(atlas_path, columns=["pair"] + KEY + ["replicate_supported"])
            atlas = atlas[atlas.pair == "Hct116_vs_K562"]
            for label, actual, expected in (
                ("selected", atlas, baseline),
                ("strict", atlas[atlas.replicate_supported], baseline[baseline.replicate_supported]),
            ):
                same = len(actual) == len(expected) and set(map(tuple, actual[KEY].to_numpy())) == set(map(tuple, expected[KEY].to_numpy()))
                record(f"atlas_baseline_{label}_regression_matches", same, "Audited tolerance convention; canonical site-set comparison")
                if not same:
                    raise ValueError(f"Atlas baseline {label} regression mismatch")
        record("baseline_matched_sites", len(d))
        record("baseline_selected", len(baseline))
        record("baseline_strict", int(baseline.replicate_supported.sum()))
        record("atlas_vs_original_baseline_extra_sites", len(baseline) - len(exact), "Atlas uses 1e-7 tolerance; original focus CSV uses exact comparisons")
        summary.to_csv(tables / "qc_differential_robustness.csv", index=False)
        robustness_figure(summary, output / "figures")
    else:
        record("differential_robustness_status", "unavailable", "Missing Hct116/K562 consensus Parquet; full matched universe cannot be reconstructed from selected-only tables")
        pd.DataFrame(columns=["grid_id", "positive_threshold", "delta_threshold", "is_baseline", "n_selected", "n_strict", "strict_fraction", "baseline_retention", "candidate_jaccard_vs_baseline"]).to_csv(tables / "qc_differential_robustness.csv", index=False)
    pd.DataFrame(overview).to_csv(tables / "qc_overview.csv", index=False)
    print(pd.DataFrame(overview).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--input-dir", "--input", dest="input_dir", type=Path, help="External prediction CSV root (or TASK2_RAW_DIR); never persisted")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--run-level-only", action="store_true", help="Update run QC and Figure S3 without regenerating differential QC")
    args = parser.parse_args()
    raw_root = args.input_dir or (Path(os.environ["TASK2_RAW_DIR"]) if os.environ.get("TASK2_RAW_DIR") else None)
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    run(args.output, raw_root, args.run_level_only, args.chunksize)


if __name__ == "__main__":
    main()
