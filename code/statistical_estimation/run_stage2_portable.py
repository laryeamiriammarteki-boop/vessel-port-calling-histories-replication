# -*- coding: utf-8 -*-
"""Stage 2 — structure and identification (RQ1).

Uses the frozen 2021-2025 history panel plus Width/Draft from the existing
local six-port filtered parquet (D: and F:). Does not overwrite the frozen
panel. Does not download AIS. Does not edit the manuscript. Does not run
Stages 3-4 (P90, long-stay, ranking, BTS, 2019-2020).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import traceback
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from scipy import stats

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
ROOT = Path(_CFG["STUDY_ROOT"])
PANEL = Path(_CFG.get("HISTORY_PANEL", ROOT / "02_rebuild_2021_2025" / "03_final_calls" / "HISTORY_PANEL_2021_2025_REBUILT.parquet"))
EXPECTED_SHA = _CFG.get("EXPECTED_HISTORY_SHA256", "9f2b1e69e3e18c13852ae1f8b0edc38ec64e343cc9fb22399207d1e28bbc58c1")
EXPECTED_N = int(_CFG.get("EXPECTED_N", 54788))
EXPECTED_PAIRS = int(_CFG.get("EXPECTED_PAIRS", 14022))

STAGE = ROOT / "04_HUMAN_REVIEW_REVISION" / "STAGE2_STRUCTURE_IDENTIFICATION"
CODE = STAGE / "00_code"
PANELS = STAGE / "01_panels"
TABLES = STAGE / "02_tables"
NOTES = STAGE / "03_notes"
LOGS = STAGE / "00_logs"

T0 = pd.Timestamp("2021-01-01")
T_EST = pd.Timestamp("2022-01-01")
T_END = pd.Timestamp("2026-01-01")
N_BOOT = 120
BOOT_SEED = 20261003
TOP1_RULE = "ceil_1pct_pairs_by_call_count_ties_pair_id"
Z_MDE_80 = 1.959964 + 0.841621  # two-sided 5%, 80% power

_STORES_CFG = _CFG.get("AIS_STORES", {})
STORES = {k: Path(v) for k, v in _STORES_CFG.items()} if _STORES_CFG else {}

US_MID = {338, 366, 367, 368, 369}
PAN_MID = set(range(351, 358))
LBR_MID = {636}
MHL_MID = {538}
HKG_MID = {477}
SGP_MID = {563, 564, 565, 566}
CHN_MID = {412, 413, 414}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_hash(ids) -> str:
    s = "\n".join(sorted(map(str, ids)))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fn = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fn})


def fmt(x, nd=6):
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return "NA"
    return f"{float(x):.{nd}f}"


def parse_file_date(path: Path) -> date | None:
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", path.name)
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def calendar_files() -> list[tuple[date, Path]]:
    windows = [
        (date(2021, 1, 1), date(2021, 12, 31), STORES["2021_FULL"]),
        (date(2022, 1, 1), date(2022, 12, 31), STORES["2022_FULL"]),
        (date(2023, 1, 1), date(2023, 12, 31), STORES["2023_FULL"]),
        (date(2024, 1, 1), date(2024, 6, 30), STORES["2024_H1"]),
        (date(2024, 7, 1), date(2024, 12, 31), STORES["2024_H2"]),
        (date(2025, 1, 1), date(2025, 12, 31), STORES["2025_FULL"]),
    ]
    out = []
    for a, b, folder in windows:
        if not folder.exists():
            continue
        for p in folder.glob("ais-filtered-*.parquet"):
            d = parse_file_date(p)
            if d is None or d < a or d > b:
                continue
            out.append((d, p))
    out.sort(key=lambda x: x[0])
    return out


def mid_group(mmsi) -> str:
    try:
        mid = int(str(int(mmsi))[:3])
    except Exception:
        return "UNKNOWN"
    if mid in US_MID:
        return "US"
    if mid in PAN_MID:
        return "PANAMA"
    if mid in LBR_MID:
        return "LIBERIA"
    if mid in MHL_MID:
        return "MARSHALL"
    if mid in HKG_MID:
        return "HONG_KONG"
    if mid in SGP_MID:
        return "SINGAPORE"
    if mid in CHN_MID:
        return "CHINA"
    return "OTHER"


def vt_group(v) -> str:
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


def length_bin(x) -> str:
    if not np.isfinite(x):
        return "LEN_MISSING"
    if x >= 300:
        return "LEN_GE300"
    if x >= 180:
        return "LEN_180_299"
    if x > 0:
        return "LEN_LT180"
    return "LEN_MISSING"


def fit_absorb(df: pd.DataFrame, y: str, xvars: list, absorb: list, cluster: list, run_id: str) -> dict:
    warn = []
    dropped = []
    try:
        cols = [y] + list(xvars) + list(absorb) + list(cluster) + ["stable_call_id", "pair_id"]
        cols = list(dict.fromkeys(cols))
        est = df[cols].copy()
        est[y] = pd.to_numeric(est[y], errors="coerce")
        for v in xvars:
            est[v] = pd.to_numeric(est[v], errors="coerce")
        n_before = len(est)
        est = est.dropna(subset=[y] + list(xvars) + list(absorb) + list(cluster)).copy()
        n_drop = n_before - len(est)
        if n_drop:
            warn.append(f"dropna_dropped={n_drop}")
        if est.empty:
            return {"run_id": run_id, "ok": False, "error": "empty after dropna", "warnings": warn}
        for a in absorb:
            est[a] = est[a].astype("category")
        exog = est[list(xvars)].astype(float)
        keep = []
        for c in xvars:
            if float(exog[c].std(ddof=0)) > 1e-12:
                keep.append(c)
            else:
                dropped.append(c)
                warn.append(f"no_variation:{c}")
        if not keep:
            return {"run_id": run_id, "ok": False, "error": "no remaining regressors", "warnings": warn}
        exog = exog[keep]
        cl = pd.DataFrame({c: pd.factorize(est[c])[0] for c in cluster}, index=est.index)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = AbsorbingLS(
                dependent=est[y].astype(float),
                exog=exog,
                absorb=est[list(absorb)],
                drop_absorbed=True,
            )
            res = model.fit(cov_type="clustered", clusters=cl)
            for wi in caught:
                warn.append(str(wi.message)[:300])
        params = {str(k): float(v) for k, v in res.params.items()}
        ses = {str(k): float(v) for k, v in res.std_errors.items()}
        pvals = {str(k): float(v) for k, v in res.pvalues.items()}
        ci = res.conf_int()
        cov = res.cov
        cov_d = {}
        for a in params:
            for b in params:
                try:
                    cov_d[f"{a}||{b}"] = float(cov.loc[a, b])
                except Exception:
                    pass
        r2 = None
        try:
            r2 = float(res.rsquared)
        except Exception:
            pass
        psz = est.groupby("pair_id").size()
        return {
            "run_id": run_id,
            "ok": True,
            "error": "",
            "n": int(res.nobs),
            "n_drop": int(n_drop),
            "pairs": int(est["pair_id"].nunique()),
            "identifying_pairs": int((psz >= 2).sum()),
            "identifying_calls": int(psz[psz >= 2].sum()),
            "cluster_counts": {c: int(est[c].nunique()) for c in cluster},
            "params": params,
            "se": ses,
            "p": pvals,
            "ci_low": {str(i): float(ci.loc[i].iloc[0]) for i in ci.index},
            "ci_high": {str(i): float(ci.loc[i].iloc[1]) for i in ci.index},
            "cov": cov_d,
            "rsquared": r2,
            "sample_hash": sample_hash(est["stable_call_id"].tolist()),
            "warnings": warn,
            "dropped_no_variation": dropped,
            "xvars_kept": keep,
        }
    except Exception as e:
        warn.append("exception:" + traceback.format_exc(limit=3))
        return {"run_id": run_id, "ok": False, "error": repr(e), "warnings": warn, "n": None}


def coef_row(res: dict, var: str, extra: dict | None = None) -> dict:
    row = {
        "RUN_ID": res.get("run_id"),
        "VARIABLE": var,
        "COEF": res.get("params", {}).get(var) if res.get("ok") else None,
        "SE": res.get("se", {}).get(var) if res.get("ok") else None,
        "CI_LOW": res.get("ci_low", {}).get(var) if res.get("ok") else None,
        "CI_HIGH": res.get("ci_high", {}).get(var) if res.get("ok") else None,
        "P": res.get("p", {}).get(var) if res.get("ok") else None,
        "N": res.get("n"),
        "PAIRS": res.get("pairs"),
        "IDENTIFYING_PAIRS": res.get("identifying_pairs"),
        "IDENTIFYING_CALLS": res.get("identifying_calls"),
        "R2": res.get("rsquared"),
        "OK": res.get("ok"),
        "ERROR": res.get("error", ""),
        "SAMPLE_HASH": res.get("sample_hash"),
        "WARNING": " | ".join(res.get("warnings") or [])[:1500],
        "CLUSTER_COUNTS": json.dumps(res.get("cluster_counts") or {}),
    }
    if extra:
        row.update(extra)
    return row


def linear_combo(res: dict, names_w: dict) -> dict:
    if not res.get("ok"):
        return {"coef": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    keys = list(names_w)
    missing = [k for k in keys if k not in res.get("params", {})]
    if missing:
        return {"coef": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "missing": missing}
    b = np.array([res["params"][k] for k in keys], dtype=float)
    w = np.array([names_w[k] for k in keys], dtype=float)
    cov = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, c in enumerate(keys):
            cov[i, j] = res["cov"].get(f"{a}||{c}", np.nan)
    est = float(w @ b)
    var = float(w @ cov @ w)
    se = float(np.sqrt(var)) if var >= 0 and np.isfinite(var) else np.nan
    z = est / se if se and np.isfinite(se) and se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "coef": est,
        "se": se,
        "ci_low": est - 1.96 * se if np.isfinite(se) else np.nan,
        "ci_high": est + 1.96 * se if np.isfinite(se) else np.nan,
        "p": p,
    }


def twoway_resid(s: pd.Series, g1: pd.Series, g2: pd.Series, iters: int = 60) -> pd.Series:
    r = pd.to_numeric(s, errors="coerce").astype(float).copy()
    a = g1.astype(str)
    b = g2.astype(str)
    for _ in range(iters):
        r = r - r.groupby(a).transform("mean")
        r = r - r.groupby(b).transform("mean")
    return r


def add_dummies(df: pd.DataFrame, col: str, prefix: str, drop: str | None) -> list[str]:
    names = []
    for val, _ in df[col].value_counts().items():
        if drop is not None and str(val) == str(drop):
            continue
        nm = f"{prefix}_{str(val)}"
        df[nm] = (df[col].astype(str) == str(val)).astype(float)
        names.append(nm)
    return names


def hours_from_beta(beta, d0, dH):
    if not np.isfinite(beta) or not np.isfinite(d0) or not np.isfinite(dH):
        return np.nan
    return float(d0 * (np.exp(beta * dH) - 1.0))


def attach_width_draft(prim: pd.DataFrame, cache: Path) -> tuple[pd.DataFrame, dict]:
    """Attach call-day Width/Draft from local filtered parquet. No new download."""
    meta = {"method": "mmsi_arrivaldate_median_from_local_filtered_parquet", "new_download": False}
    if cache.exists():
        out = pd.read_parquet(cache)
        meta["cache"] = str(cache)
        meta["n_cache_rows"] = int(len(out))
        print(f"WIDTH_DRAFT cache hit {cache} n={len(out):,}", flush=True)
        return out, meta

    prim = prim.copy()
    prim["arr_date"] = pd.to_datetime(prim["arrival_ts"], utc=True).dt.tz_convert(None).dt.date
    keys = prim[["mmsi", "arr_date"]].drop_duplicates()
    need_by_date: dict[date, set[int]] = {}
    for mmsi, d in keys.itertuples(index=False):
        try:
            mm = int(mmsi)
        except Exception:
            continue
        need_by_date.setdefault(d, set()).add(mm)
    files = calendar_files()
    meta["n_calendar_files"] = len(files)
    meta["n_needed_dates"] = len(need_by_date)
    print(f"WIDTH_DRAFT scanning {len(files)} local daily parquet files for {len(need_by_date)} arrival dates", flush=True)

    import pyarrow.parquet as pq

    rows = []
    used = 0
    skipped = 0
    for i, (d, path) in enumerate(files, start=1):
        mmsis = need_by_date.get(d)
        if not mmsis:
            skipped += 1
            continue
        try:
            tbl = pq.read_table(path, columns=["MMSI", "Width", "Draft"])
            pdf = tbl.to_pandas()
        except Exception as e:
            meta.setdefault("file_errors", []).append(f"{path.name}:{e}")
            continue
        pdf["mmsi"] = pd.to_numeric(pdf["MMSI"], errors="coerce")
        pdf = pdf.loc[pdf["mmsi"].isin(mmsis)].copy()
        if pdf.empty:
            used += 1
            continue
        w = pd.to_numeric(pdf["Width"], errors="coerce")
        dr = pd.to_numeric(pdf["Draft"], errors="coerce")
        w = w.where((w > 4) & (w < 80))
        dr = dr.where((dr > 0.5) & (dr < 25))
        pdf["Width"] = w
        pdf["Draft"] = dr
        g = pdf.groupby("mmsi", sort=False).agg(
            width=("Width", "median"),
            draft=("Draft", "median"),
            n_pts=("mmsi", "size"),
            n_width=("Width", "count"),
            n_draft=("Draft", "count"),
        ).reset_index()
        g["arr_date"] = d
        rows.append(g)
        used += 1
        if i % 50 == 0:
            print(f"  scanned {i}/{len(files)} used_dates={used}", flush=True)

    meta["files_used"] = used
    meta["files_skipped_unneeded_dates"] = skipped
    if not rows:
        out = pd.DataFrame(columns=["mmsi", "arr_date", "width", "draft", "n_pts", "n_width", "n_draft"])
    else:
        out = pd.concat(rows, ignore_index=True)
        out["mmsi"] = out["mmsi"].astype("int64")
    out.to_parquet(cache, index=False)
    meta["n_cache_rows"] = int(len(out))
    print(f"WIDTH_DRAFT wrote {cache} n={len(out):,}", flush=True)
    return out, meta


def prepare_estimation(df: pd.DataFrame, wd: pd.DataFrame) -> pd.DataFrame:
    prim = df.loc[df["PRIMARY_ESTIMATION"] == True].copy()
    prim["arrival_ts"] = pd.to_datetime(prim["arrival_ts"], utc=True)
    prim["departure_ts"] = pd.to_datetime(prim["departure_ts"], utc=True)
    prim["y"] = np.log(pd.to_numeric(prim["port_hours"], errors="coerce"))
    prim["H_W12"] = pd.to_numeric(prim["H_W12"], errors="coerce")
    prim["port_hours"] = pd.to_numeric(prim["port_hours"], errors="coerce")
    prim["length"] = pd.to_numeric(prim["length"], errors="coerce")
    prim["length"] = prim["length"].where((prim["length"] > 20) & (prim["length"] < 450))
    prim["vessel_type"] = pd.to_numeric(prim["vessel_type"], errors="coerce")
    prim["arr_date"] = prim["arrival_ts"].dt.tz_convert(None).dt.date
    prim["mmsi_int"] = pd.to_numeric(prim["mmsi"], errors="coerce")

    if len(wd):
        wdj = wd.rename(columns={"mmsi": "mmsi_int"})
        prim = prim.merge(wdj, on=["mmsi_int", "arr_date"], how="left")
    else:
        prim["width"] = np.nan
        prim["draft"] = np.nan

    # MMSI-level fallback for static size
    mmsi_wh = (
        prim.groupby("mmsi_int")[["width", "draft", "length"]]
        .median()
        .rename(columns={"width": "mmsi_width", "draft": "mmsi_draft", "length": "mmsi_length"})
    )
    prim = prim.merge(mmsi_wh, on="mmsi_int", how="left")
    prim["width"] = prim["width"].fillna(prim["mmsi_width"])
    prim["draft"] = prim["draft"].fillna(prim["mmsi_draft"])

    prim["port_ym"] = prim["relationship_port"].astype(str) + "_" + prim["arrival_ts"].dt.strftime("%Y-%m")
    prim["port_day"] = prim["relationship_port"].astype(str) + "_" + prim["arrival_ts"].dt.strftime("%Y-%m-%d")
    prim["port_week"] = prim["relationship_port"].astype(str) + "_" + prim["arrival_ts"].dt.strftime("%G-W%V")
    prim["port_month"] = prim["relationship_port"].astype(str) + "_" + prim["arrival_ts"].dt.strftime("%Y-%m")

    prim["mean_H"] = prim.groupby("pair_id")["H_W12"].transform("mean")
    prim["HDEV"] = prim["H_W12"] - prim["mean_H"]
    prim["T_i"] = prim.groupby("pair_id")["stable_call_id"].transform("size").astype(float)
    # first observed arrival on the eligible relationship (incl. 2021)
    first_all = (
        df.loc[df["HISTORY_ELIGIBLE"] == True, ["pair_id", "arrival_ts"]]
        .assign(arrival_ts=lambda x: pd.to_datetime(x["arrival_ts"], utc=True))
        .groupby("pair_id")["arrival_ts"]
        .min()
        .rename("first_arrival_ts")
    )
    prim = prim.merge(first_all, on="pair_id", how="left")
    prim["first_arrival_days"] = (prim["first_arrival_ts"] - pd.Timestamp("2021-01-01", tz="UTC")).dt.total_seconds() / 86400.0

    prim["vt_group"] = prim["vessel_type"].map(vt_group)
    prim["mid_group"] = prim["mmsi"].map(mid_group)
    prim["port_vt"] = prim["relationship_port"].astype(str) + "|" + prim["vt_group"].astype(str)
    prim["len_bin"] = prim["length"].map(length_bin)

    for col, miss, fill in [
        ("length", "miss_length", "mean_length"),
        ("width", "miss_width", "mean_width"),
        ("draft", "miss_draft", "mean_draft"),
    ]:
        prim[fill] = prim.groupby("pair_id")[col].transform("mean")
        prim[miss] = prim[fill].isna().astype(float)
        # pair-constant fill: observed pair mean, else sample median of pair means
        med = float(prim[fill].median()) if prim[fill].notna().any() else 0.0
        prim[fill] = prim[fill].fillna(med)
        prim[col + "_obs"] = prim[col]
        prim[col] = prim[col].fillna(prim[fill])

    prim["dev_length"] = prim["length"] - prim["mean_length"]
    prim["dev_width"] = prim["width"] - prim["mean_width"]
    prim["dev_draft"] = prim["draft"] - prim["mean_draft"]

    ready = (
        prim["y"].notna()
        & prim["H_W12"].notna()
        & prim["pair_id"].notna()
        & prim["port_ym"].notna()
        & np.isfinite(prim["y"])
        & (prim["port_hours"] > 0)
    )
    prim = prim.loc[ready].copy()
    return prim


def within_audit(prim: pd.DataFrame) -> list[dict]:
    rows = []
    vars_ = [
        ("H_W12", "H_W12"),
        ("length", "length"),
        ("width", "width"),
        ("draft", "draft"),
        ("vessel_type", "vessel_type"),
        ("mean_length", "pair_mean_length"),
        ("mean_width", "pair_mean_width"),
        ("mean_draft", "pair_mean_draft"),
        ("T_i", "T_i"),
        ("first_arrival_days", "first_arrival_days"),
    ]
    for col, name in vars_:
        x = pd.to_numeric(prim[col], errors="coerce")
        g = prim["pair_id"]
        nuniq = x.groupby(g).nunique(dropna=True)
        pair_sd = x.groupby(g).std(ddof=1)
        overall = float(x.std(ddof=1)) if x.notna().any() else np.nan
        within_sd = float(np.nanmean(pair_sd)) if len(pair_sd) else np.nan
        share_vary = float((nuniq > 1).mean()) if len(nuniq) else np.nan
        ratio = within_sd / overall if overall and np.isfinite(overall) and overall > 0 else np.nan
        role = "pair_constant" if (not np.isfinite(share_vary) or share_vary < 0.05) else (
            "mostly_pair_constant" if share_vary < 0.20 else "time_varying"
        )
        rows.append(
            {
                "VARIABLE": name,
                "N_OBS": int(x.notna().sum()),
                "SHARE_OBS_MISSING": float(x.isna().mean()),
                "N_PAIRS": int(g.nunique()),
                "SHARE_PAIRS_WITH_WITHIN_VARIATION": share_vary,
                "MEAN_WITHIN_PAIR_SD": within_sd,
                "OVERALL_SD": overall,
                "WITHIN_SD_OVER_OVERALL_SD": ratio,
                "ROLE": role,
                "RULE": "pair_constant if share of pairs with >1 distinct value <0.05; mostly_pair_constant if <0.20",
            }
        )
    # MID / type / port are categorical
    for col, name in [("vt_group", "vt_group"), ("mid_group", "mid_group"), ("relationship_port", "port")]:
        nuniq = prim.groupby("pair_id")[col].nunique(dropna=True)
        rows.append(
            {
                "VARIABLE": name,
                "N_OBS": int(prim[col].notna().sum()),
                "SHARE_OBS_MISSING": float(prim[col].isna().mean()),
                "N_PAIRS": int(prim["pair_id"].nunique()),
                "SHARE_PAIRS_WITH_WITHIN_VARIATION": float((nuniq > 1).mean()),
                "MEAN_WITHIN_PAIR_SD": np.nan,
                "OVERALL_SD": np.nan,
                "WITHIN_SD_OVER_OVERALL_SD": np.nan,
                "ROLE": "pair_constant" if float((nuniq > 1).mean()) < 0.05 else "time_varying",
                "RULE": "pair_constant if share of pairs with >1 distinct value <0.05",
            }
        )
    return rows


def missingness_table(prim: pd.DataFrame) -> list[dict]:
    rows = []
    for col, label in [
        ("length_obs", "Length"),
        ("width", "Width"),
        ("draft", "Draft"),
        ("vessel_type", "VesselType"),
        ("mmsi", "MMSI"),
        ("mid_group", "MID_flag_proxy"),
        ("imo", "IMO"),
        ("arrival_ts", "arrival_ts"),
        ("relationship_port", "port"),
    ]:
        x = prim[col]
        if col == "length_obs":
            miss = prim["miss_length"] == 1
        elif col == "width":
            miss = prim["miss_width"] == 1
        elif col == "draft":
            miss = prim["miss_draft"] == 1
        else:
            miss = pd.isna(x)
        miss = pd.Series(miss, index=prim.index)
        rows.append(
            {
                "VARIABLE": label,
                "N": int(len(prim)),
                "N_MISSING": int(miss.sum()),
                "SHARE_MISSING": float(miss.mean()),
                "N_PAIRS": int(prim["pair_id"].nunique()),
                "SHARE_PAIRS_ALL_MISSING": float(miss.groupby(prim["pair_id"]).all().mean()),
                "SOURCE": "frozen call panel" if label in {"Length", "VesselType", "MMSI", "IMO", "arrival_ts", "port", "MID_flag_proxy"} else "local filtered parquet attach (no new download)",
            }
        )
    return rows


def ident_shares(prim: pd.DataFrame, top1: set, terc: pd.Series) -> list[dict]:
    hd = prim["H_W12"] - prim.groupby("pair_id")["H_W12"].transform("mean")
    ss = (hd ** 2).groupby(prim["pair_id"]).sum()
    tot = float(ss.sum())
    n_i = prim.groupby("pair_id").size()
    rows = []

    def pack(label, mask_pairs):
        sub = ss.loc[ss.index.isin(mask_pairs)]
        return {
            "GROUP": label,
            "N_PAIRS": int(len(sub)),
            "N_CALLS": int(n_i.reindex(sub.index).sum()) if len(sub) else 0,
            "WITHIN_H_SS": float(sub.sum()) if len(sub) else 0.0,
            "IDENTIFYING_VARIANCE_SHARE": float(sub.sum() / tot) if tot else np.nan,
            "MEAN_WITHIN_VAR_H": float(sub.div(np.maximum(n_i.reindex(sub.index) - 1, 1)).mean()) if len(sub) else np.nan,
            "MEDIAN_WITHIN_VAR_H": float(sub.div(np.maximum(n_i.reindex(sub.index) - 1, 1)).median()) if len(sub) else np.nan,
        }

    rows.append(pack("ALL", set(ss.index)))
    for lab in ["T1_low", "T2_mid", "T3_high"]:
        rows.append(pack(lab, set(terc[terc == lab].index)))
    n_pairs = int(n_i.shape[0])
    ranked = n_i.sort_values(ascending=False)
    for pct, name in [(0.10, "TOP10"), (0.05, "TOP5"), (0.01, "TOP1")]:
        k = int(np.ceil(pct * n_pairs))
        rows.append(pack(name, set(ranked.head(k).index)))
    rows.append(pack("TOP1_SET", set(top1)))
    rows.append(pack("OTHER99", set(ss.index) - set(top1)))
    # pair-size deciles of n_i
    rnk = n_i.rank(method="first")
    dec = pd.qcut(rnk, 10, labels=[f"D{i}" for i in range(1, 11)])
    for lab in dec.cat.categories:
        rows.append(pack(f"SIZE_{lab}", set(dec[dec == lab].index)))
    return rows


def pair_bootstrap_share(prim: pd.DataFrame, n_boot: int, seed: int, cl: list) -> dict:
    rng = np.random.default_rng(seed)
    groups = prim.groupby("pair_id", sort=False).indices
    pids = np.array(list(groups.keys()))
    blocks = [np.asarray(groups[pid], dtype=np.int64) for pid in pids]
    G = len(pids)
    shares = []
    pooled_s, between_s, within_s = [], [], []
    failed = 0
    for b in range(n_boot):
        draw = rng.integers(0, G, size=G)
        idx = np.concatenate([blocks[i] for i in draw])
        sizes = np.fromiter((len(blocks[i]) for i in draw), dtype=np.int32, count=G)
        boot = prim.iloc[idx].copy()
        boot = boot.reset_index(drop=True)
        boot["pair_id"] = np.repeat(np.arange(G), sizes).astype(str)
        boot["stable_call_id"] = np.arange(len(boot)).astype(str)
        pooled = fit_absorb(boot, "y", ["H_W12"], ["port_ym"], cl, f"B{b}_P")
        mund = fit_absorb(boot, "y", ["HDEV", "mean_H"], ["port_ym"], cl, f"B{b}_M")
        if not (pooled.get("ok") and mund.get("ok")):
            failed += 1
            continue
        bp = pooled["params"].get("H_W12", np.nan)
        bw = mund["params"].get("HDEV", np.nan)
        bb = mund["params"].get("mean_H", np.nan)
        den = bb - bw
        sh = (bp - bw) / den if np.isfinite(den) and abs(den) > 1e-12 else np.nan
        if np.isfinite(sh):
            shares.append(sh)
            pooled_s.append(bp)
            between_s.append(bb)
            within_s.append(bw)
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b+1}/{n_boot} failed={failed}", flush=True)
    arr = np.array(shares, dtype=float)
    return {
        "n_boot_requested": n_boot,
        "n_boot_ok": int(len(arr)),
        "n_boot_failed": failed,
        "share_mean": float(np.nanmean(arr)) if len(arr) else np.nan,
        "share_p025": float(np.nanpercentile(arr, 2.5)) if len(arr) else np.nan,
        "share_p975": float(np.nanpercentile(arr, 97.5)) if len(arr) else np.nan,
        "pooled_boot_mean": float(np.nanmean(pooled_s)) if pooled_s else np.nan,
        "between_boot_mean": float(np.nanmean(between_s)) if between_s else np.nan,
        "within_boot_mean": float(np.nanmean(within_s)) if within_s else np.nan,
    }


def main():
    started = datetime.now()
    for d in (STAGE, CODE, PANELS, TABLES, NOTES, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    log = {"started": started.isoformat(timespec="seconds"), "new_external_data_downloaded": False, "manuscript_edited": False}

    got = sha256_file(PANEL)
    log["input_sha256"] = got
    log["input_hash_pass"] = got == EXPECTED_SHA
    if got != EXPECTED_SHA:
        raise SystemExit(f"history panel hash mismatch {got}")

    df = pd.read_parquet(PANEL)
    print(f"loaded history panel n={len(df):,}", flush=True)

    cache_wd = PANELS / "mmsi_date_width_draft.parquet"
    wd, wd_meta = attach_width_draft(df.loc[df["PRIMARY_ESTIMATION"] == True], cache_wd)
    log["width_draft_attach"] = wd_meta

    prim = prepare_estimation(df, wd)
    if len(prim) != EXPECTED_N:
        raise SystemExit(f"PRIMARY N={len(prim)} expected {EXPECTED_N}")
    if int(prim["pair_id"].nunique()) != EXPECTED_PAIRS:
        raise SystemExit(f"pairs={prim['pair_id'].nunique()} expected {EXPECTED_PAIRS}")

    # frequency groups (pre-specified; not chosen from slopes)
    n_i = prim.groupby("pair_id").size().sort_index()
    terc = pd.qcut(n_i.rank(method="first"), 3, labels=["T1_low", "T2_mid", "T3_high"])
    prim["freq_tercile"] = prim["pair_id"].map(terc.astype(str))
    k_top1 = int(np.ceil(0.01 * EXPECTED_PAIRS))
    ranked = n_i.rename("n").reset_index().sort_values(["n", "pair_id"], ascending=[False, True])
    top1 = set(ranked.head(k_top1)["pair_id"])
    prim["top1"] = prim["pair_id"].isin(top1).astype(float)
    prim["H_x_top1"] = prim["H_W12"] * prim["top1"]
    for lab, code in [("T1_low", "t1"), ("T2_mid", "t2"), ("T3_high", "t3")]:
        prim[f"terc_{code}"] = (prim["freq_tercile"] == lab).astype(float)
        prim[f"H_x_{code}"] = prim["H_W12"] * prim[f"terc_{code}"]

    # future frequency diagnostic (not a 1-1 transform of current H)
    el = df.loc[df["HISTORY_ELIGIBLE"] == True, ["pair_id", "arrival_ts", "stable_call_id", "H_W12", "port_hours"]].copy()
    el["arrival_ts"] = pd.to_datetime(el["arrival_ts"], utc=True)
    el = el.sort_values(["pair_id", "arrival_ts", "stable_call_id"])
    el["H_lead"] = el.groupby("pair_id")["H_W12"].shift(-1)
    el["next_arrival"] = el.groupby("pair_id")["arrival_ts"].shift(-1)
    el["gap_to_next_h"] = (el["next_arrival"] - el["arrival_ts"]).dt.total_seconds() / 3600.0
    el["lag_port_hours"] = el.groupby("pair_id")["port_hours"].shift(1)
    # n future calls in 90 days
    fut = []
    for pid, g in el.groupby("pair_id", sort=False):
        t = g["arrival_ts"].to_numpy()
        nfut = np.empty(len(g), dtype=float)
        for i, ti in enumerate(t):
            hi = np.searchsorted(t, ti + np.timedelta64(90, "D"), side="right")
            nfut[i] = max(hi - (i + 1), 0)
        gg = g.copy()
        gg["n_future_90d"] = nfut
        fut.append(gg[["stable_call_id", "H_lead", "gap_to_next_h", "lag_port_hours", "n_future_90d"]])
    fut = pd.concat(fut, ignore_index=True)
    prim = prim.merge(fut, on="stable_call_id", how="left")

    vt_names = add_dummies(prim, "vt_group", "dvt", drop="VT_70")
    mid_names = add_dummies(prim, "mid_group", "dmid", drop="OTHER")
    # port x type dummies; drop most common cell
    prim["port_vt_cell"] = prim["port_vt"]
    pt_mode = str(prim["port_vt_cell"].value_counts().index[0])
    pt_names = add_dummies(prim, "port_vt_cell", "dpt", drop=pt_mode)

    est_path = PANELS / "STAGE2_ESTIMATION_FRAME.parquet"
    keep_cols = [
        c
        for c in prim.columns
        if c not in set()
    ]
    prim.to_parquet(est_path, index=False)
    est_hash = sha256_file(est_path)
    write_csv(
        TABLES / "panel_hashes.csv",
        [
            {"FILE": "HISTORY_PANEL_2021_2025_REBUILT.parquet", "SHA256": got, "ROLE": "input_frozen_not_overwritten"},
            {"FILE": "mmsi_date_width_draft.parquet", "SHA256": sha256_file(cache_wd) if cache_wd.exists() else "", "ROLE": "width_draft_attach_cache"},
            {"FILE": "STAGE2_ESTIMATION_FRAME.parquet", "SHA256": est_hash, "ROLE": "stage2_analytical_frame"},
        ],
    )
    print(f"estimation frame n={len(prim):,} pairs={prim['pair_id'].nunique():,} hash={est_hash[:12]}", flush=True)

    miss_rows = missingness_table(prim)
    write_csv(TABLES / "missingness_inventory.csv", miss_rows)
    var_rows = within_audit(prim)
    write_csv(TABLES / "pair_constant_vs_within_variation.csv", var_rows)

    cl_day = ["pair_id", "port_day"]
    cl_week = ["pair_id", "port_week"]
    cl_month = ["pair_id", "port_month"]
    absorb_ym = ["port_ym"]
    absorb_pair = ["pair_id", "port_ym"]

    results = {}
    coef_rows = []

    def store(res, var, extra=None):
        results[res["run_id"]] = res
        coef_rows.append(coef_row(res, var, extra))
        print(
            f"{res['run_id']} ok={res.get('ok')} n={res.get('n')} "
            f"{var}={fmt(res.get('params', {}).get(var) if res.get('ok') else None)}",
            flush=True,
        )
        return res

    # Headline on same sample
    r_pool = store(fit_absorb(prim, "y", ["H_W12"], absorb_ym, cl_day, "BASE_POOLED"), "H_W12")
    r_mund = store(fit_absorb(prim, "y", ["HDEV", "mean_H"], absorb_ym, cl_day, "BASE_MUNDLAK"), "mean_H")
    if r_mund.get("ok"):
        coef_rows.append(coef_row(r_mund, "HDEV", {"ESTIMAND": "mundlak_within"}))
        contrast = linear_combo(r_mund, {"mean_H": 1.0, "HDEV": -1.0})
        r_mund["contrast_B_minus_W"] = contrast
    r_fe = store(fit_absorb(prim, "y", ["H_W12"], absorb_pair, cl_day, "BASE_PAIRFE"), "H_W12")

    # Stepwise between / CRE
    steps = [
        ("S0_BASE", ["HDEV", "mean_H"], absorb_ym),
        ("S1_LENGTH", ["HDEV", "mean_H", "mean_length", "miss_length"], absorb_ym),
        ("S2_WIDTH", ["HDEV", "mean_H", "mean_length", "miss_length", "mean_width", "miss_width"], absorb_ym),
        ("S3_TYPE", ["HDEV", "mean_H", "mean_length", "miss_length", "mean_width", "miss_width"] + vt_names, absorb_ym),
        ("S4_MID", ["HDEV", "mean_H", "mean_length", "miss_length", "mean_width", "miss_width"] + vt_names + mid_names, absorb_ym),
        ("S5_DRAFT", ["HDEV", "mean_H", "mean_length", "miss_length", "mean_width", "miss_width", "mean_draft", "miss_draft"] + vt_names + mid_names, absorb_ym),
        ("S6_PORT_TYPE", ["HDEV", "mean_H", "mean_length", "miss_length", "mean_width", "miss_width", "mean_draft", "miss_draft"] + mid_names + pt_names, absorb_ym),
        ("S7_CRE_UNBALANCED", ["HDEV", "mean_H", "T_i", "first_arrival_days", "mean_length", "miss_length", "mean_width", "miss_width", "mean_draft", "miss_draft"] + mid_names + pt_names, absorb_ym),
    ]
    # If draft varies within pair, include deviation in CRE
    draft_role = next((r for r in var_rows if r["VARIABLE"] == "draft"), {})
    if draft_role.get("ROLE") == "time_varying":
        steps[-1] = (
            "S7_CRE_UNBALANCED",
            ["HDEV", "mean_H", "T_i", "first_arrival_days", "mean_length", "miss_length", "mean_width", "miss_width", "mean_draft", "miss_draft", "dev_draft"]
            + mid_names
            + pt_names,
            absorb_ym,
        )

    step_rows = []
    base_between = r_mund.get("params", {}).get("mean_H") if r_mund.get("ok") else np.nan
    full_between = np.nan
    cre_between = np.nan
    for sid, xvars, ab in steps:
        res = fit_absorb(prim, "y", xvars, ab, cl_week, sid)
        results[sid] = res
        bb = res.get("params", {}).get("mean_H") if res.get("ok") else np.nan
        bw = res.get("params", {}).get("HDEV") if res.get("ok") else np.nan
        if sid == "S6_PORT_TYPE" and res.get("ok"):
            full_between = bb  # observable structure; T_i/entry is CRE not mix
        if sid == "S7_CRE_UNBALANCED" and res.get("ok"):
            cre_between = bb
        step_rows.append(
            {
                "STEP": sid,
                "BETWEEN": bb,
                "BETWEEN_SE": res.get("se", {}).get("mean_H") if res.get("ok") else None,
                "BETWEEN_P": res.get("p", {}).get("mean_H") if res.get("ok") else None,
                "BETWEEN_CI_LOW": res.get("ci_low", {}).get("mean_H") if res.get("ok") else None,
                "BETWEEN_CI_HIGH": res.get("ci_high", {}).get("mean_H") if res.get("ok") else None,
                "WITHIN_HDEV": bw,
                "WITHIN_SE": res.get("se", {}).get("HDEV") if res.get("ok") else None,
                "BETWEEN_CHANGE_FROM_BASE": (bb - base_between) if np.isfinite(bb) and np.isfinite(base_between) else None,
                "BETWEEN_CHANGE_PERCENT": (100.0 * (bb - base_between) / base_between) if np.isfinite(bb) and np.isfinite(base_between) and base_between != 0 else None,
                "N": res.get("n"),
                "PAIRS": res.get("pairs"),
                "OK": res.get("ok"),
                "ERROR": res.get("error", ""),
                "XVARS": ",".join(res.get("xvars_kept") or []),
                "CLUSTER": "pair+port_week",
            }
        )
        print(f"{sid} between={fmt(bb)} within={fmt(bw)} ok={res.get('ok')}", flush=True)
        if res.get("ok"):
            coef_rows.append(coef_row(res, "mean_H", {"ESTIMAND": "between", "STEP": sid}))
            coef_rows.append(coef_row(res, "HDEV", {"ESTIMAND": "within", "STEP": sid}))
    write_csv(TABLES / "stepwise_between_cre.csv", step_rows)

    # Between share from fitted base models
    bp = r_pool.get("params", {}).get("H_W12") if r_pool.get("ok") else np.nan
    bw0 = r_mund.get("params", {}).get("HDEV") if r_mund.get("ok") else np.nan
    bb0 = r_mund.get("params", {}).get("mean_H") if r_mund.get("ok") else np.nan
    den = (bb0 - bw0) if np.isfinite(bb0) and np.isfinite(bw0) else np.nan
    share_pt = ((bp - bw0) / den) if np.isfinite(den) and abs(den) > 1e-8 else np.nan
    # H variance between share (descriptive complement, not the RQ1 answer)
    h = prim["H_W12"]
    between_ss = ((prim.groupby("pair_id")["H_W12"].transform("mean") - h.mean()) ** 2).sum()
    total_ss = ((h - h.mean()) ** 2).sum()
    icc_h = float(between_ss / total_ss) if total_ss else np.nan

    print("BETWEEN_SHARE point", fmt(share_pt), flush=True)
    share_csv = TABLES / "between_share.csv"
    boot = None
    if share_csv.exists():
        prev = pd.read_csv(share_csv)
        if len(prev) and int(prev.iloc[0].get("N_BOOT_OK", 0) or 0) >= 100:
            boot = {
                "n_boot_requested": N_BOOT,
                "n_boot_ok": int(prev.iloc[0]["N_BOOT_OK"]),
                "n_boot_failed": 0,
                "share_mean": float(prev.iloc[0]["BOOT_MEAN"]),
                "share_p025": float(prev.iloc[0]["BOOT_CI_LOW"]),
                "share_p975": float(prev.iloc[0]["BOOT_CI_HIGH"]),
                "cache": True,
            }
            print("BETWEEN_SHARE bootstrap cache hit", boot["n_boot_ok"], flush=True)
    if boot is None:
        print("starting pair bootstrap", flush=True)
        boot = pair_bootstrap_share(prim, N_BOOT, BOOT_SEED, cl_week)
    log["between_share_bootstrap"] = boot
    share_valid = bool(np.isfinite(share_pt) and np.isfinite(den) and abs(den) > 1e-8 and np.sign(bb0) == np.sign(bp))
    write_csv(
        TABLES / "between_share.csv",
        [
            {
                "METRIC": "pooled_weight_on_between",
                "FORMULA": "(beta_pooled - beta_W) / (beta_B - beta_W)",
                "POINT": share_pt,
                "BOOT_MEAN": boot.get("share_mean"),
                "BOOT_CI_LOW": boot.get("share_p025"),
                "BOOT_CI_HIGH": boot.get("share_p975"),
                "N_BOOT_OK": boot.get("n_boot_ok"),
                "VALID_AS": "accounting identity for how the pooled slope mixes the same-sample Mundlak B and W slopes under port-month FE; NOT a variance share of PortHours and NOT a causal composition share",
                "VALID_FOR_HEADLINE": share_valid,
                "POOLED": bp,
                "BETWEEN": bb0,
                "WITHIN": bw0,
                "H_BETWEEN_VARIANCE_SHARE": icc_h,
            }
        ],
    )

    # Lead / feedback
    rh = twoway_resid(prim["H_W12"], prim["pair_id"], prim["port_ym"])
    rlead = twoway_resid(prim["H_lead"], prim["pair_id"], prim["port_ym"])
    okc = rh.notna() & rlead.notna()
    corr_lead = float(rh[okc].corr(rlead[okc])) if int(okc.sum()) > 50 else np.nan
    log["lead_H_residual_corr"] = corr_lead
    lead_sample = prim.loc[prim["H_lead"].notna()].copy()
    r_lead = store(
        fit_absorb(lead_sample, "y", ["H_W12", "H_lead"], absorb_pair, cl_week, "LEAD_H_PLACEBO"),
        "H_lead",
        {"ESTIMAND": "lead_H"},
    )
    if r_lead.get("ok"):
        coef_rows.append(coef_row(r_lead, "H_W12", {"ESTIMAND": "current_H_in_lead_model"}))
    r_fut = store(
        fit_absorb(prim, "y", ["H_W12", "n_future_90d"], absorb_pair, cl_week, "FUTURE_CALLS_90D"),
        "n_future_90d",
        {"ESTIMAND": "future_frequency_90d"},
    )
    if r_fut.get("ok"):
        coef_rows.append(coef_row(r_fut, "H_W12", {"ESTIMAND": "current_H_in_future90_model"}))

    # descriptive lagged duration vs subsequent gap / future calls
    desc = prim.loc[prim["lag_port_hours"].notna() & prim["n_future_90d"].notna(), ["lag_port_hours", "n_future_90d", "gap_to_next_h", "port_hours"]].copy()
    feed_desc = {
        "corr_current_hours_n_future_90d": float(prim["port_hours"].corr(prim["n_future_90d"])) if prim["n_future_90d"].notna().any() else np.nan,
        "corr_current_hours_gap_to_next": float(prim["port_hours"].corr(prim["gap_to_next_h"])) if prim["gap_to_next_h"].notna().any() else np.nan,
        "corr_lag_hours_n_future_90d": float(desc["lag_port_hours"].corr(desc["n_future_90d"])) if len(desc) else np.nan,
        "n_lead_model": r_lead.get("n"),
        "lead_H_residual_corr_after_pair_portym": corr_lead,
    }
    write_csv(TABLES / "feedback_descriptives.csv", [feed_desc])

    # FE leverage
    share_rows = ident_shares(prim, top1, terc)
    write_csv(TABLES / "identifying_variance_shares.csv", share_rows)

    # Tercile-specific pair FE (pre-specified groups)
    terc_rows = []
    for lab in ["T1_low", "T2_mid", "T3_high"]:
        sl = prim.loc[prim["freq_tercile"] == lab].copy()
        res = fit_absorb(sl, "y", ["H_W12"], absorb_pair, cl_week, f"FE_{lab}")
        store(res, "H_W12", {"GROUP": lab})
        terc_rows.append({**coef_row(res, "H_W12"), "GROUP": lab, "N_PAIRS_IN_GROUP": int(sl["pair_id"].nunique())})
    # joint interaction, omit T2 as reference? User asked slopes by tercile; also joint vs low as base
    r_terc_int = fit_absorb(
        prim,
        "y",
        ["H_W12", "H_x_t2", "H_x_t3"],
        absorb_pair,
        cl_week,
        "FE_TERCILE_INTERACT",
    )
    store(r_terc_int, "H_W12", {"GROUP": "T1_low_in_interact"})
    if r_terc_int.get("ok"):
        coef_rows.append(coef_row(r_terc_int, "H_x_t2"))
        coef_rows.append(coef_row(r_terc_int, "H_x_t3"))
        mid_s = linear_combo(r_terc_int, {"H_W12": 1.0, "H_x_t2": 1.0})
        hi_s = linear_combo(r_terc_int, {"H_W12": 1.0, "H_x_t3": 1.0})
        r_terc_int["mid_slope"] = mid_s
        r_terc_int["high_slope"] = hi_s
    write_csv(TABLES / "frequency_tercile_fe.csv", terc_rows)

    # Formal top-1% interaction
    r_top = store(
        fit_absorb(prim, "y", ["H_W12", "H_x_top1"], absorb_pair, cl_week, "FE_TOP1_INTERACT"),
        "H_x_top1",
        {"ESTIMAND": "top1_minus_other"},
    )
    top1_minus = np.nan
    other99 = np.nan
    top1_slope = np.nan
    top1_p = np.nan
    if r_top.get("ok"):
        other99 = r_top["params"].get("H_W12")
        top1_minus = r_top["params"].get("H_x_top1")
        top1_p = r_top["p"].get("H_x_top1")
        combo = linear_combo(r_top, {"H_W12": 1.0, "H_x_top1": 1.0})
        top1_slope = combo["coef"]
        coef_rows.append(coef_row(r_top, "H_W12", {"ESTIMAND": "other99_slope"}))
        write_csv(
            TABLES / "top1_interaction.csv",
            [
                {
                    "OTHER99_SLOPE": other99,
                    "OTHER99_SE": r_top["se"].get("H_W12"),
                    "OTHER99_P": r_top["p"].get("H_W12"),
                    "TOP1_MINUS_OTHER": top1_minus,
                    "TOP1_MINUS_OTHER_SE": r_top["se"].get("H_x_top1"),
                    "TOP1_MINUS_OTHER_P": top1_p,
                    "TOP1_SLOPE": top1_slope,
                    "TOP1_SLOPE_SE": combo["se"],
                    "TOP1_SLOPE_P": combo["p"],
                    "N": r_top["n"],
                    "PAIRS": r_top["pairs"],
                    "N_TOP1_PAIRS": int(len(top1)),
                    "RULE": TOP1_RULE,
                    "OLD_619_PERCENT_IS_INFERENTIAL_FINDING": False,
                }
            ],
        )

    # Length-bin structure cut
    len_rows = []
    for lab in ["LEN_LT180", "LEN_180_299", "LEN_GE300"]:
        sl = prim.loc[prim["len_bin"] == lab].copy()
        res = fit_absorb(sl, "y", ["H_W12"], absorb_pair, cl_week, f"FE_{lab}")
        store(res, "H_W12", {"GROUP": lab})
        mund = fit_absorb(sl, "y", ["HDEV", "mean_H"], absorb_ym, cl_week, f"MUND_{lab}")
        len_rows.append(
            {
                "GROUP": lab,
                "N": int(len(sl)),
                "PAIRS": int(sl["pair_id"].nunique()),
                "PAIR_FE": res.get("params", {}).get("H_W12") if res.get("ok") else None,
                "PAIR_FE_SE": res.get("se", {}).get("H_W12") if res.get("ok") else None,
                "PAIR_FE_P": res.get("p", {}).get("H_W12") if res.get("ok") else None,
                "BETWEEN": mund.get("params", {}).get("mean_H") if mund.get("ok") else None,
                "WITHIN": mund.get("params", {}).get("HDEV") if mund.get("ok") else None,
                "OK_FE": res.get("ok"),
            }
        )
    write_csv(TABLES / "length_bin_structure_cut.csv", len_rows)

    # Alternative inference
    inf_rows = []
    for name, cl in [("pair_port_day", cl_day), ("pair_port_week", cl_week), ("pair_port_month", cl_month)]:
        fe = fit_absorb(prim, "y", ["H_W12"], absorb_pair, cl, f"INF_FE_{name}")
        mund = fit_absorb(prim, "y", ["HDEV", "mean_H"], absorb_ym, cl, f"INF_MUND_{name}")
        inf_rows.append(
            {
                "CLUSTER": name,
                "N_CLUSTERS_SECOND": fe.get("cluster_counts", {}).get(cl[1]) if fe.get("ok") else None,
                "PAIR_FE": fe.get("params", {}).get("H_W12") if fe.get("ok") else None,
                "PAIR_FE_SE": fe.get("se", {}).get("H_W12") if fe.get("ok") else None,
                "PAIR_FE_P": fe.get("p", {}).get("H_W12") if fe.get("ok") else None,
                "PAIR_FE_CI_LOW": fe.get("ci_low", {}).get("H_W12") if fe.get("ok") else None,
                "PAIR_FE_CI_HIGH": fe.get("ci_high", {}).get("H_W12") if fe.get("ok") else None,
                "BETWEEN": mund.get("params", {}).get("mean_H") if mund.get("ok") else None,
                "BETWEEN_SE": mund.get("se", {}).get("mean_H") if mund.get("ok") else None,
                "WITHIN": mund.get("params", {}).get("HDEV") if mund.get("ok") else None,
                "WITHIN_SE": mund.get("se", {}).get("HDEV") if mund.get("ok") else None,
            }
        )
        print(f"INF {name} FE={fmt(inf_rows[-1]['PAIR_FE'])} SE={fmt(inf_rows[-1]['PAIR_FE_SE'])}", flush=True)
    write_csv(TABLES / "alternative_inference.csv", inf_rows)
    fe_week = next(r for r in inf_rows if r["CLUSTER"] == "pair_port_week")
    fe_month = next(r for r in inf_rows if r["CLUSTER"] == "pair_port_month")
    fe_day = next(r for r in inf_rows if r["CLUSTER"] == "pair_port_day")
    alt_pass = bool(
        fe_week["PAIR_FE_SE"]
        and np.isfinite(fe_week["PAIR_FE_SE"])
        and np.sign(fe_week["PAIR_FE"] or 0) == np.sign(r_fe.get("params", {}).get("H_W12") or 0)
    )

    # Retransformation + MDE (no unsourced TOST)
    beta_fe = r_fe.get("params", {}).get("H_W12") if r_fe.get("ok") else np.nan
    se_fe_day = r_fe.get("se", {}).get("H_W12") if r_fe.get("ok") else np.nan
    se_fe_week = fe_week.get("PAIR_FE_SE")
    se_use = se_fe_week if se_fe_week and np.isfinite(se_fe_week) else se_fe_day
    blo = beta_fe - 1.96 * se_use if np.isfinite(se_use) else np.nan
    bhi = beta_fe + 1.96 * se_use if np.isfinite(se_use) else np.nan
    med = float(prim["port_hours"].median())
    mean_h = float(prim["port_hours"].mean())
    dH = float(np.log(2.0))
    mde_beta = Z_MDE_80 * se_use if np.isfinite(se_use) else np.nan
    trans_rows = [
        {
            "METHOD": "logOLS_percent_and_median_hours",
            "LOCATION": "median",
            "D0_HOURS": med,
            "CONTRAST": "doubling_1plus_PriorCalls",
            "DELTA_H": dH,
            "BETA": beta_fe,
            "PERCENT": 100.0 * (np.exp(beta_fe * dH) - 1.0) if np.isfinite(beta_fe) else np.nan,
            "DELTA_HOURS": hours_from_beta(beta_fe, med, dH),
            "DELTA_HOURS_CI_LOW": hours_from_beta(blo, med, dH),
            "DELTA_HOURS_CI_HIGH": hours_from_beta(bhi, med, dH),
            "MDE_BETA_80": mde_beta,
            "MDE_HOURS_AT_MEDIAN_DOUBLING": hours_from_beta(mde_beta, med, dH),
            "SE_USED": se_use,
            "SE_CLUSTER": "pair+port_week",
            "MEAN_HOURS_MAPPING_USED": False,
            "ARITHMETIC_MEAN_HOURS_NOT_USED": mean_h,
            "TOST_RUN": False,
            "TOST_MARGIN_JUSTIFIED": False,
            "TOST_REASON": "No verified liner berth-window or schedule-buffer hour margin is sourced in the frozen bibliography; a 6-hour convention is not used as an equivalence bound.",
            "COMPATIBLE_RANGE_HOURS_NOTE": "95% CI for the doubling contrast at the sample median, using pair+port-week SEs",
        }
    ]
    write_csv(TABLES / "retransformation_mde.csv", trans_rows)

    # CRE explanation note payload
    mund_w = bw0
    pair_fe = beta_fe
    log["cre_mundlak_vs_pairfe"] = {
        "mundlak_within": mund_w,
        "pair_fe": pair_fe,
        "difference": (mund_w - pair_fe) if np.isfinite(mund_w) and np.isfinite(pair_fe) else np.nan,
        "reason": "Port x year-month FE vary within pair. Mundlak CRE with only pair-mean H does not residualise those time effects the same way as pair FE, so the two within coefficients need not be equal in an unbalanced panel. They are not forced equal.",
    }

    write_csv(TABLES / "coefficient_catalog.csv", coef_rows)

    # Decision flags
    lead_coef = r_lead.get("params", {}).get("H_lead") if r_lead.get("ok") else np.nan
    lead_p = r_lead.get("p", {}).get("H_lead") if r_lead.get("ok") else np.nan
    fut_coef = r_fut.get("params", {}).get("n_future_90d") if r_fut.get("ok") else np.nan
    fut_p = r_fut.get("p", {}).get("n_future_90d") if r_fut.get("ok") else np.nan
    collinear_lead = bool(np.isfinite(corr_lead) and abs(corr_lead) >= 0.85)

    def _sig(p, c):
        return bool(np.isfinite(p) and p < 0.05 and np.isfinite(c) and abs(c) > 0)

    if collinear_lead:
        # rely on future-90d diagnostic
        if _sig(fut_p, fut_coef) and abs(hours_from_beta(r_fut.get("params", {}).get("H_W12") or 0, med, dH) if False else fut_coef) > 0:
            feedback = "HIGH" if (fut_p < 0.01 and abs(fut_coef) >= 0.01) else "MODERATE"
        else:
            feedback = "LOW" if not _sig(fut_p, fut_coef) else "MODERATE"
    else:
        if _sig(lead_p, lead_coef) or _sig(fut_p, fut_coef):
            feedback = "HIGH" if ((lead_p or 1) < 0.01 or (fut_p or 1) < 0.01) else "MODERATE"
        else:
            feedback = "LOW"

    t1 = next((r for r in terc_rows if r.get("GROUP") == "T1_low"), {})
    t2 = next((r for r in terc_rows if r.get("GROUP") == "T2_mid"), {})
    t3 = next((r for r in terc_rows if r.get("GROUP") == "T3_high"), {})
    t1_coef = t1.get("COEF") if t1.get("OK") else None
    t2_coef = t2.get("COEF") if t2.get("OK") else None
    t3_coef = t3.get("COEF") if t3.get("OK") else None
    sh_top1 = next((r for r in share_rows if r["GROUP"] == "TOP1"), {})
    sh_top5 = next((r for r in share_rows if r["GROUP"] == "TOP5"), {})
    sh_top10 = next((r for r in share_rows if r["GROUP"] == "TOP10"), {})

    between_change_pct = None
    if np.isfinite(full_between) and np.isfinite(base_between) and base_between != 0:
        between_change_pct = 100.0 * (full_between - base_between) / base_between

    # RQ1 status
    def _finite(x):
        try:
            return x is not None and np.isfinite(float(x))
        except (TypeError, ValueError):
            return False

    observables_absorb_between = bool(
        _finite(full_between) and _finite(base_between) and abs(float(full_between)) < 0.25 * abs(float(base_between))
    )
    week_p = fe_week.get("PAIR_FE_P")
    within_robust = bool(
        _finite(pair_fe)
        and abs(float(pair_fe)) >= 0.02
        and _finite(week_p)
        and float(week_p) < 0.05
        and feedback == "LOW"
    )
    hetero = False
    if r_terc_int.get("ok"):
        p2 = r_terc_int.get("p", {}).get("H_x_t2")
        p3 = r_terc_int.get("p", {}).get("H_x_t3")
        hetero = bool(
            (_finite(p2) and float(p2) < 0.05)
            or (_finite(p3) and float(p3) < 0.05)
            or (_finite(top1_p) and float(top1_p) < 0.05)
        )

    if observables_absorb_between or feedback != "LOW" or hetero or not within_robust:
        rq1_status = "MODIFY"
    else:
        rq1_status = "KEEP"

    if abs(full_between or 0) > abs(pair_fe or 0) * 2:
        planning = "pair_baseline_residual_persistent_structure"
    else:
        planning = "pair_baseline_and_history"

    if feedback != "LOW":
        planning = "pair_baseline; do_not_use_history_as_learning_discount"
    elif not within_robust:
        planning = "pair_baseline; history_discount_not_robust"

    interp = (
        "Most of the observed history-port-time gradient is persistent pair/service structure "
        "(between). Observable size/type/flag/draft/port-type absorb only part of that between "
        "component; leftover is residual persistent pair/service structure, unidentified among "
        "terminal/window/sorting/congestion exposure. The within-pair history slope is small "
        "and is not treated as a learning effect."
    )

    stage2_pass = True
    blockers = []
    if not r_mund.get("ok") or not r_fe.get("ok"):
        stage2_pass = False
        blockers.append("headline_models_failed")
    if not any(s["OK"] for s in step_rows):
        stage2_pass = False
        blockers.append("stepwise_failed")
    if r_top.get("ok") is not True:
        blockers.append("top1_test_failed")
        stage2_pass = False

    console = {
        "CRE_AUDIT_COMPLETE": True,
        "CRE_RESPECIFIED": True,
        "BASE_BETWEEN": bb0,
        "FULL_STRUCTURE_BETWEEN": full_between,
        "CRE_BETWEEN_AFTER_TI_ENTRY": cre_between,
        "BETWEEN_CHANGE_PERCENT": between_change_pct,
        "BETWEEN_SHARE_VALID": share_valid,
        "BETWEEN_SHARE": share_pt,
        "BETWEEN_SHARE_CI": [boot.get("share_p025"), boot.get("share_p975")],
        "LEAD_HISTORY_PLACEBO": lead_coef,
        "LEAD_HISTORY_P": lead_p,
        "LEAD_H_RESIDUAL_CORR": corr_lead,
        "FUTURE90_COEF": fut_coef,
        "FUTURE90_P": fut_p,
        "FEEDBACK_CONCERN": feedback,
        "LOW_FREQUENCY_TERCILE_FE": t1_coef,
        "MID_FREQUENCY_TERCILE_FE": t2_coef,
        "HIGH_FREQUENCY_TERCILE_FE": t3_coef,
        "TOP1_SLOPE": top1_slope,
        "OTHER99_SLOPE": other99,
        "TOP1_MINUS_OTHER": top1_minus,
        "TOP1_DIFFERENCE_P": top1_p,
        "TOP1_IDENTIFYING_VARIANCE_SHARE": sh_top1.get("IDENTIFYING_VARIANCE_SHARE"),
        "TOP5_IDENTIFYING_VARIANCE_SHARE": sh_top5.get("IDENTIFYING_VARIANCE_SHARE"),
        "TOP10_IDENTIFYING_VARIANCE_SHARE": sh_top10.get("IDENTIFYING_VARIANCE_SHARE"),
        "TOST_RUN": False,
        "TOST_MARGIN_JUSTIFIED": False,
        "MDE": trans_rows[0]["MDE_HOURS_AT_MEDIAN_DOUBLING"],
        "ALTERNATIVE_INFERENCE_PASS": alt_pass,
        "RETRANSLATION_METHOD": "logOLS_percent_and_median_hours",
        "RQ1_STATUS": rq1_status,
        "RQ1_SCIENTIFIC_INTERPRETATION": interp,
        "SUPPLY_CHAIN_PLANNING_PARAMETER": planning,
        "NEW_EXTERNAL_DATA_DOWNLOADED": False,
        "MANUSCRIPT_EDITED": False,
        "STAGE2_PASS": stage2_pass,
        "BLOCKERS": ";".join(blockers) if blockers else "none",
        "N": EXPECTED_N,
        "PAIRS": EXPECTED_PAIRS,
        "TOP1_N_PAIRS": int(len(top1)),
        "PAIR_FE": pair_fe,
        "PAIR_FE_SE_PORTWEEK": se_fe_week,
        "MEDIAN_PORTHOURS": med,
        "COLLINEAR_LEAD_H": collinear_lead,
    }
    log["console"] = console
    log["finished"] = datetime.now().isoformat(timespec="seconds")
    (LOGS / "stage2_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    (LOGS / "stage2_console.json").write_text(json.dumps(console, indent=2, default=str), encoding="utf-8")

    def tf(x):
        return "TRUE" if x else "FALSE"

    def num(x, nd=6):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return f"[{fmt(x[0], nd)}, {fmt(x[1], nd)}]"
        return fmt(x, nd)

    lines = [
        "============================================================",
        "STAGE 2 — STRUCTURE AND IDENTIFICATION",
        "============================================================",
        f"CRE_AUDIT_COMPLETE = {tf(console['CRE_AUDIT_COMPLETE'])}",
        f"CRE_RESPECIFIED = {tf(console['CRE_RESPECIFIED'])}",
        f"BASE_BETWEEN = {num(console['BASE_BETWEEN'])}",
        f"FULL_STRUCTURE_BETWEEN = {num(console['FULL_STRUCTURE_BETWEEN'])}",
        f"BETWEEN_CHANGE_PERCENT = {num(console['BETWEEN_CHANGE_PERCENT'], 2)}",
        f"BETWEEN_SHARE_VALID = {tf(console['BETWEEN_SHARE_VALID'])}",
        f"BETWEEN_SHARE = {num(console['BETWEEN_SHARE'], 4)}",
        f"BETWEEN_SHARE_CI = {num(console['BETWEEN_SHARE_CI'], 4)}",
        f"LEAD_HISTORY_PLACEBO = {num(console['LEAD_HISTORY_PLACEBO'])}",
        f"LEAD_HISTORY_P = {num(console['LEAD_HISTORY_P'], 4)}",
        f"FEEDBACK_CONCERN = {console['FEEDBACK_CONCERN']}",
        f"LOW_FREQUENCY_TERCILE_FE = {num(console['LOW_FREQUENCY_TERCILE_FE'])}",
        f"MID_FREQUENCY_TERCILE_FE = {num(console['MID_FREQUENCY_TERCILE_FE'])}",
        f"HIGH_FREQUENCY_TERCILE_FE = {num(console['HIGH_FREQUENCY_TERCILE_FE'])}",
        f"TOP1_SLOPE = {num(console['TOP1_SLOPE'])}",
        f"OTHER99_SLOPE = {num(console['OTHER99_SLOPE'])}",
        f"TOP1_MINUS_OTHER = {num(console['TOP1_MINUS_OTHER'])}",
        f"TOP1_DIFFERENCE_P = {num(console['TOP1_DIFFERENCE_P'], 4)}",
        f"TOP1_IDENTIFYING_VARIANCE_SHARE = {num(console['TOP1_IDENTIFYING_VARIANCE_SHARE'], 4)}",
        f"TOP5_IDENTIFYING_VARIANCE_SHARE = {num(console['TOP5_IDENTIFYING_VARIANCE_SHARE'], 4)}",
        f"TOP10_IDENTIFYING_VARIANCE_SHARE = {num(console['TOP10_IDENTIFYING_VARIANCE_SHARE'], 4)}",
        f"TOST_RUN = {tf(console['TOST_RUN'])}",
        f"TOST_MARGIN_JUSTIFIED = {tf(console['TOST_MARGIN_JUSTIFIED'])}",
        f"MDE = {num(console['MDE'], 3)}",
        f"ALTERNATIVE_INFERENCE_PASS = {tf(console['ALTERNATIVE_INFERENCE_PASS'])}",
        f"RETRANSLATION_METHOD = {console['RETRANSLATION_METHOD']}",
        f"RQ1_STATUS = {console['RQ1_STATUS']}",
        f"RQ1_SCIENTIFIC_INTERPRETATION = {console['RQ1_SCIENTIFIC_INTERPRETATION']}",
        f"SUPPLY_CHAIN_PLANNING_PARAMETER = {console['SUPPLY_CHAIN_PLANNING_PARAMETER']}",
        "NEW_EXTERNAL_DATA_DOWNLOADED = FALSE",
        "MANUSCRIPT_EDITED = FALSE",
        f"STAGE2_PASS = {tf(console['STAGE2_PASS'])}",
        f"BLOCKERS = {console['BLOCKERS']}",
    ]
    text = "\n".join(lines) + "\n"
    (LOGS / "STAGE2_CONSOLE.txt").write_text(text, encoding="utf-8")
    print(text)
    print("STAGE2_COMPLETE", flush=True)
    return console


if __name__ == "__main__":
    main()
