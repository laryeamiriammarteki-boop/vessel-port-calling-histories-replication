# -*- coding: utf-8 -*-
"""Gate C4B: forecast and matched-scheduling sensitivity on existing episode panels.

Reads frozen panels and the Gate 2D executable rules. Writes only inside this folder.
Does not modify the manuscript, supplement, prior audit files, or source panels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# PORTABLE PATH BLOCK (path configuration only; scientific constants unchanged)
import json as _json, os as _os
def _load_paths():
    env = _os.environ.get("TP_PATHS_CONFIG")
    cands = [Path(env)] if env else []
    cands += [
        Path(__file__).resolve().parents[2] / "paths_config.local.json",
        Path(r"F:/DATA/TP_VESSEL_PORT_CALLING_HISTORIES_REPLICATION/local_repo_staging/paths_config.local.json"),
    ]
    for p in cands:
        if p and Path(p).is_file():
            return _json.loads(Path(p).read_text(encoding="utf-8"))
    raise RuntimeError("paths_config.local.json not found")
_CFG = _load_paths()
OUT = Path(_CFG["C4B_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUT)
print("WORKING DIRECTORY", os.getcwd(), flush=True)

ROOT = Path(_CFG["STUDY_ROOT"])
REV = (
    ROOT / "04_HUMAN_REVIEW_REVISION" / "07_REVIEWER_RESPONSE_REVISION"
    / "13_WORKING_REVISION_FROM_RECEIVED_REVIEW"
)
CODE = Path(_CFG.get("REREVIEW_CODE", REV / "00_code"))
STAGE1 = ROOT / "04_HUMAN_REVIEW_REVISION" / "STAGE1_CONSTRUCTOR_VALIDITY" / "01_panels"
STAGE2 = Path(_CFG["STAGE2_FRAME"])
HIST = Path(_CFG["HISTORY_PANEL"])
G06 = Path(_CFG["CALLS_G06"])
G12 = Path(_CFG["CALLS_G12"])
MERGE12 = Path(_CFG["CALLS_MERGE12"])
CONSTRUCTOR = ROOT / "01_code" / "rebuild" / "constructor_v1.py"
STAGE1_CODE = (
    ROOT / "04_HUMAN_REVIEW_REVISION" / "STAGE1_CONSTRUCTOR_VALIDITY"
    / "00_code" / "run_stage1.py"
)
GATE2D = (
    REV / "14_SCIENTIFIC_IMPROVEMENT_PROGRAMME_20261009" / "03_METHODS_RESULTS_AUDIT"
    / "GATE2D_OPTIONB_FORECAST_P90_REANALYSIS_20261009" / "_run_gate2d_reanalysis.py"
)
MANUSCRIPT = Path(_CFG.get("MANUSCRIPT_PLACEHOLDER", OUT / "_NO_MANUSCRIPT_IN_PORTABLE_COPY"))
SUPPLEMENT = Path(_CFG.get("SUPPLEMENT_PLACEHOLDER", OUT / "_NO_SUPPLEMENT_IN_PORTABLE_COPY"))
BTS = Path(_CFG["BTS_CSV"])
BTS_NOTE = (
    ROOT / "04_HUMAN_REVIEW_REVISION" / "STAGE4_PLANNING_CONSEQUENCE"
    / "03_notes" / "BTS_CPPI_POLICY_CONTEXT.md"
)
INTERMEDIATE = OUT / "intermediate"
SNAPSHOTS = OUT / "00_SOURCE_SNAPSHOTS"

ORIGIN0 = pd.Timestamp("2022-07-01", tz="UTC")
T2022 = pd.Timestamp("2022-01-01", tz="UTC")
LAM_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
ALPHA = 0.3
NBOOT_MAE = 1000
NBOOT_P90M = 400
SEED_MAE = 20261006
SEED_P90M = 20261009
SEED_COMMON_MAE = 20261016
SEED_COMMON_P90 = 20261019
PUBLISHED_DIFF = -2.5
STORED_NESTED_MAE = 33.574678929836445
STORED_CELL_MAE = 36.06736534759864
STORED_DIFF = -2.4926864177621866
STORED_TW_LO = -2.938994406132629
STORED_TW_HI = -1.974626737652029
STORED_ICC_WINDOW = 119.19423786859973
STORED_CELL_WINDOW = 120.06188549627372
STORED_SLACK = -3.9982609193981964
STORED_SLACK_LO = -7.930842299196008
STORED_SLACK_HI = 0.9048855976180071
STORED_OV_ICC = 0.10398554575406525
STORED_OV_CELL = 0.10477202678286747
STORED_N4_SLACK = -8.789602241513142
STORED_N4_LO = -15.433677880945988
STORED_N4_HI = 0.424984509963719
STORED_N4_OV_ICC = 0.10224243374627567
STORED_N4_OV_CELL = 0.10369295907166379
STORED_N4_N = 25508
STORED_RHO = 0.639980552399452

sys.path.insert(0, str(CODE))
from run_rereview_round3 import demean  # noqa: E402
from run_rereview_round5 import icc_of, shrink_med  # noqa: E402
from run_rereview_round8 import cell_lookup  # noqa: E402
from run_rereview_round14 import calib_scale, slack_block  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def length_bin(x) -> str:
    """Stage 2 length class (run_stage2.py)."""
    if not np.isfinite(x):
        return "LEN_MISSING"
    if x >= 300:
        return "LEN_GE300"
    if x >= 180:
        return "LEN_180_299"
    if x > 0:
        return "LEN_LT180"
    return "LEN_MISSING"


def vt_group(v) -> str:
    """Stage 2 vessel-type class (run_stage2.py)."""
    try:
        x = int(v)
    except Exception:
        return "VT_UNKNOWN"
    if x == 70:
        return "VT_70"
    if 71 <= x <= 74:
        return "VT_71_74"
    if 75 <= x <= 79:
        return "VT_75_79"
    return "VT_OTHER"


def recode_vt(v):
    """Gate 2D history-only fallback."""
    if pd.isna(v):
        return "VT_70"
    x = float(v)
    if x == 70:
        return "VT_70"
    if 71 <= x <= 74:
        return "VT_71_74"
    if 75 <= x <= 79:
        return "VT_75_79"
    return "VT_70"


def recode_len(L):
    """Gate 2D history-only fallback. Not identical to length_bin at nonpositive length."""
    if pd.isna(L):
        return "LEN_MISSING"
    x = float(L)
    if x < 180:
        return "LEN_LT180"
    if x < 300:
        return "LEN_180_299"
    return "LEN_GE300"


def boot_mae_contrasts(sl, rng, nboot):
    tot_p = sl.groupby("pair").agg(dsum=("d", "sum"), nn=("d", "size"))
    dsum, nn = tot_p["dsum"].to_numpy(), tot_p["nn"].to_numpy()
    pair_boot = []
    for _ in range(nboot):
        ix = rng.choice(len(dsum), size=len(dsum), replace=True)
        pair_boot.append(float(dsum[ix].sum() / nn[ix].sum()))
    bym = sl.groupby("MONTH")["d"].mean().to_numpy()
    month_boot = [float(rng.choice(bym, size=len(bym), replace=True).mean()) for _ in range(nboot)]
    pm = sl.groupby(["MONTH", "pair"]).agg(dsum=("d", "sum"), nn=("d", "size")).reset_index()
    months = pm["MONTH"].unique()
    grouped = {m: pm.loc[pm["MONTH"] == m] for m in months}
    tw_boot = []
    for _ in range(nboot):
        md = rng.choice(months, size=len(months), replace=True)
        ds = ns = 0.0
        for m in md:
            g = grouped[m]
            ix = rng.choice(len(g), size=len(g), replace=True)
            ds += float(g["dsum"].to_numpy()[ix].sum())
            ns += float(g["nn"].to_numpy()[ix].sum())
        tw_boot.append(ds / ns if ns else np.nan)
    return {
        "MAE_DIFF": float(sl["d"].mean()),
        "PAIR_P025": float(np.percentile(pair_boot, 2.5)),
        "PAIR_P975": float(np.percentile(pair_boot, 97.5)),
        "MONTH_P025": float(np.percentile(month_boot, 2.5)),
        "MONTH_P975": float(np.percentile(month_boot, 97.5)),
        "TWOWAY_P025": float(np.nanpercentile(tw_boot, 2.5)),
        "TWOWAY_P975": float(np.nanpercentile(tw_boot, 97.5)),
    }


def pair_boot_diff(d, pair, rng, nboot):
    gp = pd.DataFrame({"d": d, "pair": pair}).groupby("pair")["d"].agg(["sum", "size"])
    dsum, nn = gp["sum"].to_numpy(), gp["size"].to_numpy()
    boots = []
    for _ in range(nboot):
        ix = rng.choice(len(dsum), size=len(dsum), replace=True)
        boots.append(float(dsum[ix].sum() / nn[ix].sum()))
    return float(np.mean(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def attach_completed_pair_stats(s2, hist, alpha=ALPHA):
    n_s = len(s2)
    n_call = np.zeros(n_s, dtype=np.int32)
    n_2021 = np.zeros(n_s, dtype=np.int32)
    med = np.full(n_s, np.nan)
    p90 = np.full(n_s, np.nan)
    pos = {cid: i for i, cid in enumerate(s2["stable_call_id"].astype(str))}
    t2022 = np.datetime64("2022-01-01T00:00:00")
    for pair, gh in hist.groupby("pair_id", sort=False):
        arr = pd.to_datetime(gh["arrival_ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        dep = pd.to_datetime(gh["departure_ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        hrs = pd.to_numeric(gh["port_hours"], errors="coerce").to_numpy()
        cids = gh["stable_call_id"].astype(str).to_numpy()
        for i in range(len(gh)):
            j = pos.get(cids[i])
            if j is None:
                continue
            t = arr[i]
            done = (dep < t) & np.isfinite(hrs)
            k = int(done.sum())
            n_call[j] = k
            n_2021[j] = int((done & (arr < t2022)).sum())
            if k == 0:
                continue
            hh = hrs[done]
            med[j] = float(np.median(hh))
            if k >= 4:
                p90[j] = float(np.quantile(hh, 0.90))
    s2 = s2.copy()
    s2["n_call"] = n_call.astype(float)
    s2["n_2021_completed"] = n_2021
    s2["pair_med_exp"] = med
    s2["pair_p90_exp"] = p90
    return s2


def month_start(m):
    return pd.Timestamp(str(m) + "-01", tz="UTC")


def val_months(s2_months, m):
    prior = [x for x in s2_months if x < m]
    if len(prior) >= 8:
        return list(prior[-6:])
    return list(prior[-max(len(prior) - 2, 1):])


def flag_true(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def canon_token(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    rounded = num.round()
    as_int = num.notna() & np.isclose(num.to_numpy(dtype=float), rounded.to_numpy(dtype=float), equal_nan=False)
    out = s.astype(str).str.strip()
    out = out.where(~as_int, rounded.astype("Int64").astype(str))
    return out


def episode_key(df: pd.DataFrame) -> pd.Series:
    arr = pd.to_datetime(df["arrival_ts"], utc=True).astype("int64")
    dep = pd.to_datetime(df["departure_ts"], utc=True).astype("int64")
    return (
        canon_token(df["mmsi"]) + "|"
        + canon_token(df["portid"]) + "|"
        + arr.astype(str) + "|"
        + dep.astype(str)
    )


def prepare_clock(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["arrival_ts"] = pd.to_datetime(out["arrival_ts"], utc=True)
    out["departure_ts"] = pd.to_datetime(out["departure_ts"], utc=True)
    out["pair_id"] = out["pair_id"].astype(str)
    out["stable_call_id"] = out["stable_call_id"].astype(str)
    out["port_hours"] = pd.to_numeric(out["port_hours"], errors="coerce")
    out["month"] = out["arrival_ts"].dt.tz_localize(None).dt.to_period("M").astype(str)
    return out


def inherit_attributes(s2: pd.DataFrame, el: pd.DataFrame) -> pd.DataFrame:
    """Pair attributes follow the Gate 2D history fill.

    Primary rows keep their own class. History rows of a primary pair inherit the
    first primary row. History-only pairs use the Gate 2D recode fallback.
    """
    el = el.copy()
    el = el.drop(columns=[c for c in ("vt_group", "len_bin") if c in el.columns])
    pair_attr = s2.groupby("pair_id", sort=False)[["vt_group", "len_bin", "relationship_port"]].first()
    el = el.merge(pair_attr, on="pair_id", how="left", suffixes=("", "_s2"))
    miss = el["vt_group"].isna()
    el.loc[miss, "vt_group"] = el.loc[miss, "vessel_type"].map(recode_vt)
    el.loc[miss, "len_bin"] = el.loc[miss, "length"].map(recode_len)
    if "relationship_port_s2" in el.columns:
        el["relationship_port"] = el["relationship_port"].fillna(el["relationship_port_s2"])
    el["vt_group"] = el["vt_group"].astype(str)
    el["len_bin"] = el["len_bin"].astype(str)
    el["port_ym"] = el["relationship_port"].astype(str) + "_" + el["month"]
    return el


def assign_stage2_bins(s2: pd.DataFrame) -> pd.DataFrame:
    s2 = s2.copy()
    length = pd.to_numeric(s2["length"], errors="coerce").to_numpy(dtype=float)
    s2["len_bin"] = [length_bin(v) for v in length]
    s2["vt_group"] = [vt_group(v) for v in s2["vessel_type"].to_numpy()]
    return s2


def load_b0():
    s2 = prepare_clock(pd.read_parquet(STAGE2))
    s2["vt_group"] = s2["vt_group"].astype(str)
    s2["len_bin"] = s2["len_bin"].astype(str)
    length = pd.to_numeric(s2["length"], errors="coerce").to_numpy(dtype=float)
    recomputed = [length_bin(v) for v in length]
    n_len_disagree = int(sum(a != b for a, b in zip(recomputed, s2["len_bin"].tolist())))
    hist = pd.read_parquet(HIST)
    el = prepare_clock(hist.loc[flag_true(hist["HISTORY_ELIGIBLE"])].copy())
    el = inherit_attributes(s2, el)
    meta = {
        "n_primary": int(len(s2)),
        "n_history": int(len(el)),
        "n_2021": int((el["arrival_ts"] < T2022).sum()),
        "n_len_bin_not_function_of_current_length": n_len_disagree,
        "len_bin_rule": "frozen STAGE2 len_bin column; not recomputed",
        "vt_group_rule": "frozen STAGE2 vt_group column",
        "history_fill": "first primary pair attributes; Gate 2D recode_vt/recode_len if pair absent",
    }
    return s2, el, meta


def load_alternative(path: Path, construction: str):
    raw = pd.read_parquet(path)
    required = [
        "stable_call_id", "pair_id", "port_hours", "arrival_ts", "departure_ts",
        "relationship_port", "vessel_type", "length", "mmsi", "portid", "imo",
        "HISTORY_ELIGIBLE", "PRIMARY_ESTIMATION",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise SystemExit(f"BLOCKER {construction} missing columns: {missing}")
    el = prepare_clock(raw.loc[flag_true(raw["HISTORY_ELIGIBLE"])].copy())
    s2 = prepare_clock(raw.loc[flag_true(raw["PRIMARY_ESTIMATION"])].copy())
    s2 = assign_stage2_bins(s2)
    el = inherit_attributes(s2, el)
    meta = {
        "n_rows": int(len(raw)),
        "n_primary": int(len(s2)),
        "n_history": int(len(el)),
        "n_2021": int((el["arrival_ts"] < T2022).sum()),
        "n_fragments_gt1": int((raw["n_fragments"] > 1).sum()) if "n_fragments" in raw.columns else 0,
        "gap_h_values": sorted(pd.to_numeric(raw["gap_h"], errors="coerce").dropna().unique().tolist()) if "gap_h" in raw.columns else [],
        "sog_kn_values": sorted(pd.to_numeric(raw["sog_kn"], errors="coerce").dropna().unique().tolist()) if "sog_kn" in raw.columns else [],
        "len_bin_rule": "stage2 length_bin applied to the panel length field on primary rows",
        "vt_group_rule": "stage2 vt_group applied to the panel vessel_type field on primary rows",
        "history_fill": "first primary pair attributes; Gate 2D recode_vt/recode_len if pair absent",
    }
    return s2, el, meta, raw


def run_protocol(s2: pd.DataFrame, el: pd.DataFrame, construction: str):
    if s2["stable_call_id"].duplicated().any() or el["stable_call_id"].duplicated().any():
        raise SystemExit(f"BLOCKER {construction}: stable_call_id is not unique")
    missing_from_history = set(s2["stable_call_id"]) - set(el["stable_call_id"])
    if missing_from_history:
        raise SystemExit(
            f"BLOCKER {construction}: {len(missing_from_history)} primary calls are absent from history"
        )
    s2 = s2.sort_values(["pair_id", "arrival_ts", "stable_call_id"]).reset_index(drop=True)
    s2["is_forecast"] = s2["arrival_ts"] >= ORIGIN0
    print(construction, "attach pair stats", len(s2), len(el), flush=True)
    s2 = attach_completed_pair_stats(s2, el)
    fc = s2.loc[s2["is_forecast"]]
    leak = int((np.isfinite(fc["pair_p90_exp"]) & (fc["n_call"] < 4)).sum())
    if leak:
        raise SystemExit(f"BLOCKER {construction}: pair P90 present with n<4 ({leak})")
    if int(((fc["n_call"] == 0) & np.isfinite(fc["pair_med_exp"])).sum()):
        raise SystemExit(f"BLOCKER {construction}: cold-start pair median is finite")
    s2_months = sorted(s2["month"].unique())
    origins = [m for m in s2_months if m >= "2022-07"]
    july_val = val_months(s2_months, "2022-07") if "2022-07" in origins else []
    print(construction, "origins", len(origins), "july_val", july_val, flush=True)
    rows = []
    chosen = []
    n4_fallback_calls = 0
    for m in origins:
        Tm = month_start(m)
        test = s2.loc[s2["month"] == m].copy()
        train = el.loc[el["departure_ts"] < Tm].copy()
        if train.empty or test.empty:
            continue
        if (train["departure_ts"] >= Tm).any():
            raise SystemExit(f"BLOCKER {construction}: training departure on or after {m}")
        vms = val_months(s2_months, m)
        inner = train.loc[~train["month"].isin(vms)]
        val = s2.loc[s2["month"].isin(vms)].copy()
        val = val.loc[val["departure_ts"] < Tm]
        if inner.empty or val.empty:
            inner, val = train, train
            vms = ["FALLBACK_FULL_TRAIN"]
        grand_in = float(inner["port_hours"].median())
        fine_in = inner.groupby(["relationship_port", "vt_group", "len_bin"])["port_hours"].median()
        n_in = val["n_call"].to_numpy()
        pmed_in = val["pair_med_exp"].to_numpy()
        ptf_in = cell_lookup(
            list(zip(val["relationship_port"], val["vt_group"], val["len_bin"])), fine_in, grand_in
        )
        yv_in = val["port_hours"].to_numpy()
        best_lam_f, best_mae_f = 1.0, np.inf
        for lam_try in LAM_GRID:
            pred_f = shrink_med(n_in, np.where(np.isfinite(pmed_in), pmed_in, ptf_in), ptf_in, lam_try)
            mae_f = float(np.nanmean(np.abs(yv_in - pred_f)))
            if np.isfinite(mae_f) and mae_f < best_mae_f:
                best_mae_f, best_lam_f = mae_f, lam_try
        ytr = demean(np.log(train["port_hours"]), train["port_ym"])
        _, lam_icc = icc_of(ytr, train["pair_id"])
        if not np.isfinite(lam_icc) or lam_icc <= 0:
            lam_icc = 1.0
        pt_fine = train.groupby(["relationship_port", "vt_group", "len_bin"])["port_hours"].median()
        p90_fine = train.groupby(["relationship_port", "vt_group", "len_bin"])["port_hours"].quantile(0.90)
        grand = float(train["port_hours"].median())
        gp90 = float(train["port_hours"].quantile(0.90))
        n_call = test["n_call"].to_numpy()
        pair_med = test["pair_med_exp"].to_numpy()
        pair_p90 = test["pair_p90_exp"].to_numpy()
        ptf = cell_lookup(
            list(zip(test["relationship_port"], test["vt_group"], test["len_bin"])), pt_fine, grand
        )
        ptf90 = cell_lookup(
            list(zip(test["relationship_port"], test["vt_group"], test["len_bin"])), p90_fine, gp90
        )
        yv = test["port_hours"].to_numpy()
        nest_f = shrink_med(n_call, np.where(np.isfinite(pair_med), pair_med, ptf), ptf, best_lam_f)
        p90_in = inner.groupby(["relationship_port", "vt_group", "len_bin"])["port_hours"].quantile(0.90)
        gp_in = float(inner["port_hours"].quantile(0.90))
        ytr_in = demean(np.log(inner["port_hours"]), inner["port_ym"])
        _, lam_in = icc_of(ytr_in, inner["pair_id"])
        if not np.isfinite(lam_in) or lam_in <= 0:
            lam_in = 1.0
        v_n = val["n_call"].to_numpy()
        v_pp = val["pair_p90_exp"].to_numpy()
        v_cell = cell_lookup(
            list(zip(val["relationship_port"], val["vt_group"], val["len_bin"])), p90_in, gp_in
        )
        v_w = v_n / (v_n + lam_in)
        v_icc = np.where((v_n >= 4) & np.isfinite(v_pp), v_w * v_pp + (1.0 - v_w) * v_cell, v_cell)
        v_y = val["port_hours"].to_numpy()
        c_icc = calib_scale(v_icc, v_y, 0.10)
        c_cell = calib_scale(v_cell, v_y, 0.10)
        m4 = v_n >= 4
        n4_ok = int(m4.sum()) >= 30
        if n4_ok:
            c_icc_n4 = calib_scale(v_icc[m4], v_y[m4], 0.10)
            c_cell_n4 = calib_scale(v_cell[m4], v_y[m4], 0.10)
        else:
            c_icc_n4, c_cell_n4 = c_icc, c_cell
            n4_fallback_calls += int(len(test))
        w_icc = n_call / (n_call + lam_icc)
        p90_icc = np.where((n_call >= 4) & np.isfinite(pair_p90), w_icc * pair_p90 + (1.0 - w_icc) * ptf90, ptf90)
        rows.append(pd.DataFrame({
            "MONTH": m,
            "y": yv,
            "n": n_call,
            "pair": test["pair_id"].to_numpy(),
            "cid": test["stable_call_id"].astype(str).to_numpy(),
            "pred_nested": nest_f,
            "pred_base": ptf,
            "p90_cell": ptf90,
            "p90_icc": p90_icc,
            "c_icc": c_icc,
            "c_cell": c_cell,
            "c_icc_n4": c_icc_n4,
            "c_cell_n4": c_cell_n4,
            "lam_icc": lam_icc,
            "nested_lam": best_lam_f,
            "imo": canon_token(test["imo"]).to_numpy(),
            "relationship_port": test["relationship_port"].astype(str).to_numpy(),
            "episode_key": episode_key(test).to_numpy(),
            "n_fragments": test["n_fragments"].to_numpy() if "n_fragments" in test.columns else np.ones(len(test)),
        }))
        chosen.append({
            "construction": construction,
            "MONTH": m,
            "NESTED_LAM_FINE": best_lam_f,
            "VAL_MAE": best_mae_f,
            "LAM_ICC_TRAIN": lam_icc,
            "LAM_ICC_INNER": lam_in,
            "C_ICC": c_icc,
            "C_CELL": c_cell,
            "C_ICC_N4": c_icc_n4,
            "C_CELL_N4": c_cell_n4,
            "N4_VAL": int(m4.sum()),
            "N4_CALIBRATION_FALLBACK": (not n4_ok),
            "VAL_MONTHS": "|".join(vms),
            "N_TRAIN": int(len(train)),
            "N_INNER": int(len(inner)),
            "N_VAL": int(len(val)),
            "N_TEST": int(len(test)),
        })
        print(construction, m, "lam", best_lam_f, "n_test", len(test), flush=True)
    pred = pd.concat(rows, ignore_index=True)
    pred["construction"] = construction
    pred["icc_roll"] = pred["c_icc"] * pred["p90_icc"]
    pred["cell_roll"] = pred["c_cell"] * pred["p90_cell"]
    pred["icc_n4"] = pred["c_icc_n4"] * pred["p90_icc"]
    pred["cell_n4"] = pred["c_cell_n4"] * pred["p90_cell"]
    info = {
        "july_validation_months": july_val,
        "origin_months": origins,
        "n_origins_with_tests": int(pred["MONTH"].nunique()),
        "n4_fallback_forecast_calls": int(n4_fallback_calls),
        "first_origin": origins[0] if origins else None,
        "last_origin": origins[-1] if origins else None,
    }
    return pred, pd.DataFrame(chosen), info


def n_distribution(n: pd.Series) -> dict:
    n = n.astype(float)
    return {
        "n0": int((n == 0).sum()),
        "n1": int((n == 1).sum()),
        "n2": int((n == 2).sum()),
        "n3": int((n == 3).sum()),
        "nge4": int((n >= 4).sum()),
        "n_p50": float(n.median()),
        "n_p90": float(n.quantile(0.90)),
        "n_mean": float(n.mean()),
    }


def coverage_comparable(r_icc, r_cell) -> bool:
    """Predeclared reporting label. Not a new calibration target."""
    if not (np.isfinite(r_icc) and np.isfinite(r_cell)):
        return False
    return (0.08 <= r_icc <= 0.12) and (0.08 <= r_cell <= 0.12) and abs(r_icc - r_cell) < 0.01


def forecast_row(pred: pd.DataFrame, construction: str, scope: str, boot: dict, extra: dict) -> dict:
    y = pred["y"].to_numpy(float)
    ae_n = np.abs(y - pred["pred_nested"].to_numpy(float))
    ae_b = np.abs(y - pred["pred_base"].to_numpy(float))
    mae_n = float(ae_n.mean())
    mae_b = float(ae_b.mean())
    diff = float((ae_n - ae_b).mean())
    rel = diff / mae_b if mae_b else np.nan
    row = {
        "construction": construction,
        "comparison_scope": scope,
        "comparison_type": "within_construction_nested_minus_length_class_baseline",
        "absolute_mae_not_comparable_across_constructions": True,
        "forecast_call_count": int(len(pred)),
        "vessel_port_relationships": int(pred["pair"].nunique()),
        "baseline_mae_hours": mae_b,
        "nested_mae_hours": mae_n,
        "paired_mae_difference_hours": diff,
        "paired_mae_difference_twoway_p025": boot.get("TWOWAY_P025"),
        "paired_mae_difference_twoway_p975": boot.get("TWOWAY_P975"),
        "paired_mae_difference_pair_p025": boot.get("PAIR_P025"),
        "paired_mae_difference_pair_p975": boot.get("PAIR_P975"),
        "paired_mae_difference_month_p025": boot.get("MONTH_P025"),
        "paired_mae_difference_month_p975": boot.get("MONTH_P975"),
        "relative_mae_change": rel,
        "published_benchmark_difference_hours": PUBLISHED_DIFF,
        "stored_gate2d_difference_hours": STORED_DIFF,
        "difference_minus_published_minus_2_5": diff - PUBLISHED_DIFF,
        "difference_minus_stored_gate2d": diff - STORED_DIFF,
        "headline_interval": "pair_by_origin_month_two_way",
        "bootstrap_draws": boot.get("draws"),
        "bootstrap_seed": boot.get("seed"),
        "reason_if_not_computed": "",
    }
    row.update(n_distribution(pred["n"]))
    row.update(extra)
    return row


def scheduling_block(pred, construction, scope, sample, buffer_icc, buffer_cell, rng, seed, label):
    y = pred["y"].to_numpy(float)
    pr = pred["pair"].to_numpy()
    icc = pred[buffer_icc].to_numpy(float)
    cell = pred[buffer_cell].to_numpy(float)
    icc_row = slack_block(y, icc, pr, sample, buffer_icc)
    cell_row = slack_block(y, cell, pr, sample, buffer_cell)
    d_u = np.maximum(icc - y, 0) - np.maximum(cell - y, 0)
    mu, ulo, uhi = pair_boot_diff(d_u, pr, rng, NBOOT_P90M)
    comparable = coverage_comparable(icc_row["SHARE_OVERRUN"], cell_row["SHARE_OVERRUN"])
    return {
        "construction": construction,
        "comparison_scope": scope,
        "comparison_type": "within_construction_icc_pair_rule_minus_length_class_rule",
        "sample": sample,
        "protocol": label,
        "evaluation_call_count": int(len(pred)),
        "vessel_port_relationships": int(pred["pair"].nunique()),
        "mean_calibrated_allowance_icc_hours": icc_row["MEAN_WINDOW"],
        "mean_calibrated_allowance_cell_hours": cell_row["MEAN_WINDOW"],
        "mean_calibration_scale_icc": float(pred["c_icc_n4" if buffer_icc == "icc_n4" else "c_icc"].mean()),
        "mean_calibration_scale_cell": float(pred["c_cell_n4" if buffer_cell == "cell_n4" else "c_cell"].mean()),
        "overrun_incidence_icc": icc_row["SHARE_OVERRUN"],
        "overrun_incidence_cell": cell_row["SHARE_OVERRUN"],
        "overrun_incidence_difference_icc_minus_cell": icc_row["SHARE_OVERRUN"] - cell_row["SHARE_OVERRUN"],
        "mean_unused_slack_icc_hours": icc_row["MEAN_UNUSED_ALL"],
        "mean_unused_slack_cell_hours": cell_row["MEAN_UNUSED_ALL"],
        "unused_slack_difference_hours": mu,
        "unused_slack_difference_p025": ulo,
        "unused_slack_difference_p975": uhi,
        "uncertainty": "pair_bootstrap_calibration_held_fixed",
        "bootstrap_draws": NBOOT_P90M,
        "bootstrap_seed": seed,
        "mean_overrun_hours_icc": icc_row["MEAN_OVERRUN_ALL"],
        "mean_overrun_hours_cell": cell_row["MEAN_OVERRUN_ALL"],
        "mean_overrun_hours_difference_icc_minus_cell": icc_row["MEAN_OVERRUN_ALL"] - cell_row["MEAN_OVERRUN_ALL"],
        "overrun_severity_interval": "not_in_headline_matched_procedure",
        "approximately_comparable_overrun_coverage": comparable,
        "unused_slack_labeled_matched_coverage_benefit": comparable,
        "coverage_rule": "both overrun shares in [0.08, 0.12] and absolute difference < 0.01",
        "target_overrun_on_training_validation": 0.10,
        "calibration_fit_on_evaluation_outcomes": False,
        "reason_if_not_computed": "",
    }


def blank_forecast(construction, scope, reason, extra=None):
    row = {
        "construction": construction,
        "comparison_scope": scope,
        "comparison_type": "within_construction_nested_minus_length_class_baseline",
        "absolute_mae_not_comparable_across_constructions": True,
        "forecast_call_count": np.nan,
        "reason_if_not_computed": reason,
        "published_benchmark_difference_hours": PUBLISHED_DIFF,
    }
    if extra:
        row.update(extra)
    return row


def unambiguous_common(left: pd.DataFrame, right: pd.DataFrame, right_name: str):
    """Exact episode keys only. Merged or duplicate keys are not forced."""
    note = []
    r = right
    if right_name == "B2":
        ambiguous_fragments = int((r["n_fragments"].astype(float) > 1).sum())
        note.append(f"B2 forecast rows with n_fragments>1 excluded from matching: {ambiguous_fragments}")
        r = r.loc[r["n_fragments"].astype(float) == 1].copy()
    else:
        ambiguous_fragments = 0
    l_dup = int(left["episode_key"].duplicated(keep=False).sum())
    r_dup = int(r["episode_key"].duplicated(keep=False).sum())
    if l_dup or r_dup:
        note.append(f"duplicate keys excluded: left {l_dup}, right {r_dup}")
    l = left.loc[~left["episode_key"].duplicated(keep=False)].copy()
    r = r.loc[~r["episode_key"].duplicated(keep=False)].copy()
    m = l.merge(r, on="episode_key", suffixes=("_b0", "_alt"))
    if m.empty:
        return m, "; ".join(note + ["no exact key intersection"])
    same_hours = np.isclose(m["y_b0"], m["y_alt"], rtol=0, atol=1e-6)
    same_imo = m["imo_b0"].astype(str) == m["imo_alt"].astype(str)
    same_port = m["relationship_port_b0"].astype(str) == m["relationship_port_alt"].astype(str)
    note.append(
        f"exact timestamp keys {len(m)}; hours differ {int((~same_hours).sum())}; "
        f"imo differ {int((~same_imo).sum())}; relationship_port differ {int((~same_port).sum())}"
    )
    keep = m.loc[same_hours & same_imo & same_port].copy()
    note.append(f"unambiguous common forecast calls {len(keep)}")
    return keep, "; ".join(note)


def side_frame(common, suffix):
    cols = {
        "y": f"y_{suffix}",
        "n": f"n_{suffix}",
        "pair": f"pair_{suffix}",
        "MONTH": f"MONTH_{suffix}",
        "pred_nested": f"pred_nested_{suffix}",
        "pred_base": f"pred_base_{suffix}",
        "icc_roll": f"icc_roll_{suffix}",
        "cell_roll": f"cell_roll_{suffix}",
        "icc_n4": f"icc_n4_{suffix}",
        "cell_n4": f"cell_n4_{suffix}",
        "c_icc": f"c_icc_{suffix}",
        "c_cell": f"c_cell_{suffix}",
        "c_icc_n4": f"c_icc_n4_{suffix}",
        "c_cell_n4": f"c_cell_n4_{suffix}",
    }
    out = pd.DataFrame({k: common[v].to_numpy() for k, v in cols.items()})
    return out


def verify_g06_against_stage2(s2: pd.DataFrame) -> dict:
    g = pd.read_parquet(G06, columns=[
        "stable_call_id", "port_hours", "PRIMARY_ESTIMATION", "arrival_ts", "departure_ts", "mmsi", "portid"
    ])
    gp = g.loc[flag_true(g["PRIMARY_ESTIMATION"])].copy()
    a = set(s2["stable_call_id"].astype(str))
    b = set(gp["stable_call_id"].astype(str))
    merged = s2[["stable_call_id", "port_hours"]].merge(
        gp[["stable_call_id", "port_hours"]], on="stable_call_id", suffixes=("_s2", "_g06")
    )
    max_abs = float(np.nanmax(np.abs(
        pd.to_numeric(merged["port_hours_s2"], errors="coerce")
        - pd.to_numeric(merged["port_hours_g06"], errors="coerce")
    ))) if len(merged) else np.nan
    s2_keys = set(episode_key(s2))
    g_keys = set(episode_key(gp))
    return {
        "g06_primary_n": int(len(gp)),
        "id_intersection": int(len(a & b)),
        "ids_only_stage2": int(len(a - b)),
        "ids_only_g06": int(len(b - a)),
        "port_hours_max_abs_diff": max_abs,
        "episode_key_intersection": int(len(s2_keys & g_keys)),
    }


def bts_check() -> dict:
    raw = pd.read_csv(BTS)
    note = BTS_NOTE.read_text(encoding="utf-8", errors="replace") if BTS_NOTE.exists() else ""
    return {
        "frozen_csv_sha256": sha256_file(BTS),
        "frozen_csv_bytes": BTS.stat().st_size,
        "n_rows": int(len(raw)),
        "columns": list(raw.columns),
        "month_min": str(raw["Month"].iloc[0]) if "Month" in raw.columns else None,
        "month_max": str(raw["Month"].iloc[-1]) if "Month" in raw.columns else None,
        "stored_spearman_not_recomputed_for_revision": STORED_RHO,
        "series_modified": False,
        "local_note_mentions_120": "120" in note,
        "local_note_mentions_168": "168" in note,
        "local_note_sha256": sha256_file(BTS_NOTE) if BTS_NOTE.exists() else None,
    }


def write_freeze(hashes, b0_meta, g06_check, july_val):
    lines = [
        "# C4B source and method freeze",
        "",
        "Estimation started only after the checks in this file passed.",
        "No received manuscript, supplement, prior Comment 4 audit file, or source panel was modified.",
        "",
        f"Working directory: `{OUT}`",
        "",
        "## Software and seeds",
        "",
        f"- Python {sys.version}",
        f"- numpy {np.__version__}",
        f"- pandas {pd.__version__}",
        f"- platform {platform.platform()}",
        f"- Headline forecast bootstrap seed {SEED_MAE}, draws {NBOOT_MAE}. The generator is used only for the nested-rule versus length-class contrast, matching the first Gate 2D contrast on this seed.",
        f"- Matched scheduling bootstrap seed {SEED_P90M}, draws {NBOOT_P90M}. Consumption order copies Gate 2D: overall roll10, overall n4-calibration, n>=4 roll10, n>=4 n4-calibration, then n<4 and n=0.",
        "- Seeds 20261007 and 20261008 are the original by-history and unmatched-buffer seeds. This gate does not consume them.",
        f"- Common-call forecast bootstrap seed {SEED_COMMON_MAE}; common-call scheduling seed {SEED_COMMON_P90}. These seeds apply only to the unambiguous subset.",
        "- No new random sampling of calls. Pair medians use the deterministic numpy median and quantile.",
        "",
        "## Executable being reproduced",
        "",
        f"- Gate 2D script: `{GATE2D}`",
        "- Helpers imported, not rewritten: `demean`, `icc_of`, `shrink_med`, `cell_lookup`, `calib_scale`, `slack_block`.",
        "- Forecast origins: calendar months from July 2022 through December 2025 that contain primary calls.",
        "- Evaluation arrivals are on or after 2022-07-01 UTC. History eligible for training includes 2021.",
        "- A training row for origin month m must have departure strictly before the first instant of m.",
        "- Pair predictors on a focal call use only history rows with departure strictly before that call's arrival. Same-month earlier completed stays can enter the pair median. Cell medians, lambda, and calibration constants use only the pre-origin training window.",
        "- Validation months are the last six chronologically sorted primary months before the origin when at least eight such months exist, and otherwise the last max(n-2, 1). July 2022 therefore uses four months.",
        f"- Verified July 2022 validation months on B0 before estimation: {july_val}.",
        "- Lambda grid {0.25, 0.5, 1, 2, 4, 8, 16}. Selection minimises validation MAE. The test month is not used to select lambda.",
        "- Nested prediction: w = n/(n+lambda); n>0 uses w times the completed-pair median plus (1-w) times the port x vessel-type x length-class median; n=0 uses the cell median.",
        "- Baseline is that cell median.",
        "- ICC lambda for the reported P90 weight is estimated on the full pre-origin training window. The calibration scale is estimated on the inner-window ICC and on inner-window cell P90 values, each scaled to a 10% overrun share on validation calls that have already departed. This is the stored executable asymmetry.",
        "- Pair P90 is used only when completed history is at least 4. Otherwise the cell P90 is used.",
        "- Subgroup n>=4 constants are fit on validation calls with n>=4 when that validation count is at least 30; otherwise the overall constants are used.",
        "- Look-ahead scales fit on evaluation outcomes are not computed and are not reported.",
        "- Headline forecast interval resamples origin months and, within each draw, vessel-port pairs.",
        "- Headline scheduling interval resamples vessel-port pairs and holds the fitted scales fixed.",
        "",
        "## Length class",
        "",
        f"- B0 keeps the frozen stage2 `len_bin` and `vt_group`. Recomputing `length_bin` on the current `length` column disagrees on {b0_meta['n_len_bin_not_function_of_current_length']} primary calls. Those stored labels are not replaced.",
        "- B1 and B2 primary rows use `length_bin` and `vt_group` from `run_stage2.py` on the panel's own `length` and `vessel_type`. The alternative panels do not contain the frozen `len_bin` column or `length_obs`.",
        "- History rows for a pair that appears in the primary frame inherit the first primary row's class. History-only pairs use the Gate 2D `recode_vt` / `recode_len` fallback.",
        "",
        "## Coverage label, declared before alternative results",
        "",
        "Overrun coverage is called approximately comparable only when both observed evaluation overrun shares are inside [0.08, 0.12] and their absolute difference is below 0.01.",
        "If that label is false, the unused-slack difference is still reported and is not called a matched-coverage benefit.",
        "The 10% figure remains the training-validation target. It is not refit on the evaluation sample.",
        "",
        "## Populations",
        "",
        "- B0 is the six-hour interruption benchmark. Forecasting uses the stage2 estimation frame plus the rebuilt 2021-2025 history panel, which is the Gate 2D input pair.",
        f"- G06 primary versus stage2: {json.dumps(g06_check)}.",
        "- B1 is the existing twelve-hour interruption panel `CALLS_G12_S05.parquet`. It is not combined with merging.",
        "- B2 is the existing merge panel `CALLS_MERGE12.parquet`: consecutive history-eligible same-pair episodes with departure-to-arrival gap below 12 hours are already stitched. Duration on a stitched episode is the stored first-arrival to last-departure span. This gate does not rebuild it.",
        "- B1 and B2 are estimated separately.",
        "",
        "## Input hashes",
        "",
        "| Role | SHA256 | Bytes | Path |",
        "|---|---|---:|---|",
    ]
    for row in hashes:
        lines.append(f"| {row['role']} | `{row['sha256']}` | {row['bytes']} | `{row['path']}` |")
    lines += [
        "",
        "B0 reproduction targets, used as a stop check before B1 and B2:",
        f"- Nested MAE {STORED_NESTED_MAE}",
        f"- Length-class MAE {STORED_CELL_MAE}",
        f"- Paired difference {STORED_DIFF}",
        f"- ICC allowance {STORED_ICC_WINDOW}; cell allowance {STORED_CELL_WINDOW}",
        f"- Unused-slack difference {STORED_SLACK}",
        f"- n>=4 subgroup unused-slack difference {STORED_N4_SLACK} on N={STORED_N4_N}",
        "",
    ]
    (OUT / "C4B_SOURCE_AND_METHOD_FREEZE.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    inputs = [
        ("manuscript", MANUSCRIPT),
        ("supplement", SUPPLEMENT),
        ("stage2_frame", STAGE2),
        ("history_panel", HIST),
        ("calls_g06", G06),
        ("calls_g12", G12),
        ("calls_merge12", MERGE12),
        ("gate2d_script", GATE2D),
        ("constructor_v1", CONSTRUCTOR),
        ("stage1_runner", STAGE1_CODE),
        ("round3", CODE / "run_rereview_round3.py"),
        ("round5", CODE / "run_rereview_round5.py"),
        ("round8", CODE / "run_rereview_round8.py"),
        ("round14", CODE / "run_rereview_round14.py"),
        ("bts_nfsh_p62e", BTS),
    ]
    hashes = []
    for role, path in inputs:
        if not path.exists():
            raise SystemExit(f"BLOCKER missing input: {path}")
        hashes.append({
            "role": role, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
        })
        if path.suffix == ".py":
            shutil.copy2(path, SNAPSHOTS / path.name)
    pd.DataFrame(hashes).to_csv(INTERMEDIATE / "C4B_INPUT_HASHES.csv", index=False)

    print("verify B0", flush=True)
    s2, el, b0_meta = load_b0()
    if b0_meta["n_primary"] != 54788 or b0_meta["n_history"] != 68779 or b0_meta["n_2021"] != 13991:
        raise SystemExit(f"BLOCKER B0 population {b0_meta}")
    g06_check = verify_g06_against_stage2(s2)
    if (
        g06_check["ids_only_stage2"]
        or g06_check["ids_only_g06"]
        or g06_check["port_hours_max_abs_diff"] != 0
        or g06_check["episode_key_intersection"] != 54788
    ):
        raise SystemExit(f"BLOCKER G06 is not the stage2 six-hour population: {g06_check}")
    s2_months = sorted(s2["month"].unique())
    july_val = val_months(s2_months, "2022-07")
    if july_val != ["2022-03", "2022-04", "2022-05", "2022-06"]:
        raise SystemExit(f"BLOCKER July 2022 validation months {july_val}")
    if [m for m in s2_months if m >= "2022-07"] != pd.period_range("2022-07", "2025-12", freq="M").astype(str).tolist():
        raise SystemExit("BLOCKER B0 origin calendar is not July 2022 through December 2025")
    write_freeze(hashes, b0_meta, g06_check, july_val)
    print("freeze written; B0 estimation starts", flush=True)

    constructions = {}
    s2_b0, el_b0 = s2, el
    pred0, chosen0, info0 = run_protocol(s2_b0, el_b0, "B0")
    if len(pred0) != 47045:
        raise SystemExit(f"BLOCKER B0 forecast N {len(pred0)}")
    mae_n = float(np.mean(np.abs(pred0["y"] - pred0["pred_nested"])))
    mae_b = float(np.mean(np.abs(pred0["y"] - pred0["pred_base"])))
    diff = float(np.mean(np.abs(pred0["y"] - pred0["pred_nested"]) - np.abs(pred0["y"] - pred0["pred_base"])))
    icc_w = float(pred0["icc_roll"].mean())
    cell_w = float(pred0["cell_roll"].mean())
    slack = float((np.maximum(pred0["icc_roll"] - pred0["y"], 0) - np.maximum(pred0["cell_roll"] - pred0["y"], 0)).mean())
    print("B0 points", mae_n, mae_b, diff, icc_w, cell_w, slack, flush=True)
    point_ok = (
        abs(mae_n - STORED_NESTED_MAE) < 1e-6
        and abs(mae_b - STORED_CELL_MAE) < 1e-6
        and abs(diff - STORED_DIFF) < 1e-6
        and abs(icc_w - STORED_ICC_WINDOW) < 1e-4
        and abs(cell_w - STORED_CELL_WINDOW) < 1e-4
        and abs(slack - STORED_SLACK) < 1e-4
    )
    (INTERMEDIATE / "C4B_B0_POINT_CHECK.json").write_text(json.dumps({
        "mae_nested": mae_n, "mae_cell": mae_b, "diff": diff,
        "icc_window": icc_w, "cell_window": cell_w, "slack": slack,
        "point_ok": point_ok,
    }, indent=2), encoding="utf-8")
    if not point_ok:
        raise SystemExit("BLOCKER B0 point estimates do not reproduce Gate 2D. B1 and B2 were not estimated.")
    constructions["B0"] = (pred0, chosen0, info0, b0_meta)

    print("verify B1 B2", flush=True)
    s2_1, el_1, meta_1, raw_1 = load_alternative(G12, "B1")
    if meta_1["n_rows"] != 67938 or meta_1["n_history"] != 66725 or meta_1["n_primary"] != 53142:
        raise SystemExit(f"BLOCKER B1 population {meta_1}")
    if meta_1["gap_h_values"] != [12] or meta_1["sog_kn_values"] != [0.5]:
        raise SystemExit(f"BLOCKER B1 constructor stamps {meta_1}")
    s2_2, el_2, meta_2, raw_2 = load_alternative(MERGE12, "B2")
    if meta_2["n_rows"] != 65941 or meta_2["n_history"] != 65941 or meta_2["n_primary"] != 52467:
        raise SystemExit(f"BLOCKER B2 population {meta_2}")
    if "n_fragments" not in raw_2.columns:
        raise SystemExit("BLOCKER B2 has no n_fragments column")
    constructions["B1"] = (*run_protocol(s2_1, el_1, "B1"), meta_1)
    constructions["B2"] = (*run_protocol(s2_2, el_2, "B2"), meta_2)

    forecast_rows = []
    schedule_rows = []
    primary_pairs = {
        "B0": int(s2_b0["pair_id"].nunique()),
        "B1": int(s2_1["pair_id"].nunique()),
        "B2": int(s2_2["pair_id"].nunique()),
    }
    for name, (pred, chosen, info, meta) in constructions.items():
        pred.to_parquet(INTERMEDIATE / f"C4B_{name}_PREDICTIONS.parquet", index=False)
        chosen.to_csv(INTERMEDIATE / f"C4B_{name}_ORIGIN_TUNING.csv", index=False)
        print(name, "bootstrap", len(pred), flush=True)
        sl = pd.DataFrame({
            "d": np.abs(pred["y"] - pred["pred_nested"]) - np.abs(pred["y"] - pred["pred_base"]),
            "pair": pred["pair"].to_numpy(),
            "MONTH": pred["MONTH"].to_numpy(),
        })
        boot = boot_mae_contrasts(sl, np.random.default_rng(SEED_MAE), NBOOT_MAE)
        boot["draws"] = NBOOT_MAE
        boot["seed"] = SEED_MAE
        extra = {
            "primary_call_count": meta["n_primary"],
            "history_eligible_count": meta["n_history"],
            "history_2021_count": meta["n_2021"],
            "primary_vessel_port_relationships": primary_pairs[name],
            "july_validation_months": "|".join(info["july_validation_months"]),
            "origin_count": info["n_origins_with_tests"],
            "first_origin": info["first_origin"],
            "last_origin": info["last_origin"],
            "n4_fallback_forecast_calls": info["n4_fallback_forecast_calls"],
        }
        forecast_rows.append(forecast_row(pred, name, "full_evaluation_sample", boot, extra))
        rng = np.random.default_rng(SEED_P90M)
        schedule_rows.append(scheduling_block(
            pred, name, "full_evaluation_sample", "overall", "icc_roll", "cell_roll",
            rng, SEED_P90M, "training_10pct_overall_constants",
        ))
        schedule_rows.append(scheduling_block(
            pred, name, "full_evaluation_sample", "overall", "icc_n4", "cell_n4",
            rng, SEED_P90M, "training_10pct_n4_constants_on_full_sample_rng_continuation",
        ))
        nge4 = pred.loc[pred["n"] >= 4].copy()
        schedule_rows.append(scheduling_block(
            nge4, name, "full_evaluation_sample", "n_at_least_4", "icc_roll", "cell_roll",
            rng, SEED_P90M, "overall_constants_applied_to_n_at_least_4",
        ))
        schedule_rows.append(scheduling_block(
            nge4, name, "full_evaluation_sample", "n_at_least_4", "icc_n4", "cell_n4",
            rng, SEED_P90M, "subgroup_training_10pct_constants",
        ))
        # consume the same trailing RNG draws as Gate 2D so the n>=4 interval matches
        for sample_name, mask in (("n_below_4", pred["n"] < 4), ("n0", pred["n"] == 0)):
            sub = pred.loc[mask]
            schedule_rows.append(scheduling_block(
                sub, name, "rng_continuation_not_headline", sample_name, "icc_roll", "cell_roll",
                rng, SEED_P90M, "rng_order_only",
            ))
            schedule_rows.append(scheduling_block(
                sub, name, "rng_continuation_not_headline", sample_name, "icc_n4", "cell_n4",
                rng, SEED_P90M, "rng_order_only",
            ))

    # Fix primary pair counts, which were not on pred.attrs.
    primary_pairs = {"B0": int(s2_b0["pair_id"].nunique()), "B1": int(s2_1["pair_id"].nunique()), "B2": int(s2_2["pair_id"].nunique())}
    for row in forecast_rows:
        row["primary_vessel_port_relationships"] = primary_pairs[row["construction"]]

    # Common-call subset. Each construction is compared with itself on the shared keys.
    common_notes = []
    b0_pred = constructions["B0"][0]
    for alt_name in ("B1", "B2"):
        common, note = unambiguous_common(b0_pred, constructions[alt_name][0], alt_name)
        common_notes.append({"alternative": alt_name, "note": note, "n_common": int(len(common))})
        if common.empty:
            forecast_rows.append(blank_forecast(alt_name, f"common_calls_with_B0", note))
            forecast_rows.append(blank_forecast("B0", f"common_calls_with_{alt_name}", note))
            continue
        for suffix, cname in (("b0", "B0"), ("alt", alt_name)):
            side = side_frame(common, suffix)
            sl = pd.DataFrame({
                "d": np.abs(side["y"] - side["pred_nested"]) - np.abs(side["y"] - side["pred_base"]),
                "pair": side["pair"].to_numpy(),
                "MONTH": side["MONTH"].to_numpy(),
            })
            boot = boot_mae_contrasts(sl, np.random.default_rng(SEED_COMMON_MAE), NBOOT_MAE)
            boot["draws"] = NBOOT_MAE
            boot["seed"] = SEED_COMMON_MAE
            forecast_rows.append(forecast_row(
                side, cname, f"unambiguous_common_calls_B0_and_{alt_name}", boot,
                {"common_call_note": note, "primary_call_count": np.nan},
            ))
            rng = np.random.default_rng(SEED_COMMON_P90)
            schedule_rows.append(scheduling_block(
                side, cname, f"unambiguous_common_calls_B0_and_{alt_name}", "overall",
                "icc_roll", "cell_roll", rng, SEED_COMMON_P90,
                "training_10pct_overall_constants_on_common_calls",
            ))
            nge4 = side.loc[side["n"] >= 4].copy()
            if len(nge4) >= 30:
                schedule_rows.append(scheduling_block(
                    nge4, cname, f"unambiguous_common_calls_B0_and_{alt_name}", "n_at_least_4",
                    "icc_n4", "cell_n4", rng, SEED_COMMON_P90,
                    "subgroup_training_10pct_constants_on_common_calls",
                ))
            else:
                schedule_rows.append({
                    "construction": cname,
                    "comparison_scope": f"unambiguous_common_calls_B0_and_{alt_name}",
                    "sample": "n_at_least_4",
                    "reason_if_not_computed": f"common n>=4 count {len(nge4)} below reporting floor 30",
                })
    pd.DataFrame(common_notes).to_csv(INTERMEDIATE / "C4B_COMMON_CALL_MATCH_NOTES.csv", index=False)

    # Crosswalk
    cross = []
    specs = [
        ("B0", "six-hour interruption benchmark", G06, "gap > 6 hours starts a new episode", "no merge", b0_meta, s2_b0, constructions["B0"][0]),
        ("B1", "twelve-hour interruption, existing panel", G12, "gap > 12 hours starts a new episode", "no merge", meta_1, s2_1, constructions["B1"][0]),
        ("B2", "merge consecutive same-pair episodes with gap < 12 hours, existing panel", MERGE12, "built from the six-hour episodes", "departure-to-arrival gap < 12 hours; stored span", meta_2, s2_2, constructions["B2"][0]),
    ]
    hash_by_path = {h["path"]: h for h in hashes}
    for cid, label, path, interrupt, merge_rule, meta, primary, pred in specs:
        h = hash_by_path[str(path)]
        cross.append({
            "construction": cid,
            "label": label,
            "source_file": str(path),
            "sha256": h["sha256"],
            "bytes": h["bytes"],
            "interruption_rule": interrupt,
            "merge_rule": merge_rule,
            "sog_max_kn": 0.5,
            "minimum_episode_hours": 3,
            "vessel_types": "70-79",
            "combined_with_other_alternative": False,
            "n_primary": meta["n_primary"],
            "n_history_eligible": meta["n_history"],
            "n_history_2021": meta["n_2021"],
            "n_primary_vessel_port_pairs": int(primary["pair_id"].nunique()),
            "n_forecast_calls": int(len(pred)),
            "n_forecast_vessel_port_pairs": int(pred["pair"].nunique()),
            "forecast_minus_B0_calls": int(len(pred) - len(constructions["B0"][0])),
            "n_fragments_gt1": meta.get("n_fragments_gt1", 0),
            "len_bin_rule": meta["len_bin_rule"],
            "g06_stage2_id_match": g06_check["id_intersection"] if cid == "B0" else "",
            "g06_stage2_hours_max_abs_diff": g06_check["port_hours_max_abs_diff"] if cid == "B0" else "",
        })
    pd.DataFrame(cross).to_csv(OUT / "C4B_PANEL_CONSTRUCTION_CROSSWALK.csv", index=False)
    pd.DataFrame(forecast_rows).to_csv(OUT / "C4B_FORECAST_SENSITIVITY_RESULTS.csv", index=False)
    pd.DataFrame(schedule_rows).to_csv(OUT / "C4B_SCHEDULING_SENSITIVITY_RESULTS.csv", index=False)
    (INTERMEDIATE / "C4B_BTS_CORRELATION_CHECK.json").write_text(
        json.dumps(bts_check(), indent=2), encoding="utf-8"
    )
    (INTERMEDIATE / "C4B_RUN_STATUS.json").write_text(json.dumps({
        "b0_point_ok": True,
        "constructions": list(constructions),
        "common_notes": common_notes,
    }, indent=2), encoding="utf-8")
    print("ESTIMATION_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
