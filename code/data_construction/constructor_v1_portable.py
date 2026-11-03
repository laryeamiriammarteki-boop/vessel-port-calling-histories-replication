# -*- coding: utf-8 -*-
"""
Gate 2 frozen rebuild constructor (versioned copy).

Do not overwrite historical extract_mc_calls.py / extract_port_calls_pilot.py.
This module is the only constructor the 2021-2025 rebuild may call.

Frozen from the historically executed production pipeline:
  extract_mc_calls.py argparse --min-hours 3.0 --sog-max 0.5
  -> extract_port_calls_pilot.points_to_calls(..., gap_hours=6.0)
  AIS_BASELINE_EXTRACTION_SPEC.md (25 August 2026)
  R6 Methods: SOG <= 0.5; missing SOG above threshold; gap 6 h; min 3 h;
  first-point IMO; duplicates not dropped; MMSI x gateway segmentation.

M3 resolution: min_hours=3.0 is the executed production rule, not 4.0.
The 4.0 value is only the unused function-signature default on the pilot.

M9 resolution: SAME_CONSTRUCTOR_DIFFERENT_PATH for historical H1.
This rebuild uses the six-port parquet path for every year, including 2024 H1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frozen configuration (audit block)
# ---------------------------------------------------------------------------
CONSTRUCTOR_VERSION = "rebuild_constructor_v1"
SELECTED_COLUMNS_BY_NAME = [
    "MMSI",
    "BaseDateTime",
    "LAT",
    "LON",
    "SOG",
    "VesselType",
    "VesselName",
    "IMO",
    "Length",
    "portid",
    "portname",
]
CARGO_VESSELTYPE_MIN = 70
CARGO_VESSELTYPE_MAX = 79
SOG_MAX_KN = 0.5
SOG_MIN_KN = 0.0
SOG_AIS_NA_SENTINEL = 102.2
GAP_HOURS = 6.0
MIN_HOURS = 3.0
SEGMENT_ENTITY = ("portid", "mmsi")
SORT_KEYS = ("portid", "mmsi", "ts")
DUPLICATE_POLICY = "retain_sort_by_time"
IMO_POLICY = "first_point_of_segment_after_normalisation"
IMO_REQUIRED_AT = "call_stage_for_estimation_not_point_stage"
MMSI_FALLBACK_FOR_IMO = False
SAN_PEDRO_PORTID = "port664"
LALB_SPLIT_LON = -118.245
LALB_STATISTIC = "mean_longitude_all_filtered_points_in_call_window"
# 72_extract_wait_berth.py: avg(lon) over filtered AIS points with
# mmsi+portid and ts in [arrival_ts, departure_ts]; then
# mean_lon < -118.245 -> Los_Angeles else Long_Beach.
GATEWAY_ASSIGNMENT = "use_preassigned_portid_from_filtered_parquet"
PARTITION_BOUNDARY_POLICY = "segment_continuously_across_day_month_halfyear_year"

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
PROJECT_ROOT = Path(_CFG["STUDY_ROOT"])
MANIFEST_PATH = Path(_CFG.get("SOURCE_MANIFEST", PROJECT_ROOT / "02_SOURCE_MANIFEST.csv"))
REBUILD_ROOT = PROJECT_ROOT / "02_rebuild_2021_2025"


def normalize_imo(series: pd.Series) -> pd.Series:
    """Call-stage IMO normalisation. Does not invent values or MMSI fallback."""
    s = series.astype("string").str.strip()
    s = s.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
            "NAT": pd.NA,
            "NaT": pd.NA,
            "0": pd.NA,
            "0.0": pd.NA,
            "IMO0": pd.NA,
            "IMO0000000": pd.NA,
        }
    )
    s = s.where(~s.str.upper().isin(["NAN", "NONE", "<NA>", "NAT", "NULL"]), other=pd.NA)
    return s


def sog_eligible(sog: pd.Series) -> pd.Series:
    """Missing / AIS-unavailable SOG is not eligible as low-speed.

    NA and SOG >= 102.2 (ITU-R M.1371 not-available) are excluded.
    Sentinel values are not coerced to zero.
    Equivalent to historical sog.fillna(99) <= 0.5 on observed data
    (Gate 1 found 0 negative SOG); negatives would also be excluded here.
    """
    x = pd.to_numeric(sog, errors="coerce")
    return x.notna() & (x >= SOG_MIN_KN) & (x <= SOG_MAX_KN) & (x < SOG_AIS_NA_SENTINEL)


def load_parquet_by_name(path: Path) -> pd.DataFrame:
    """Load constructor fields by NAME only. Positional column access is forbidden."""
    df = pd.read_parquet(path, columns=list(SELECTED_COLUMNS_BY_NAME))
    missing = [c for c in SELECTED_COLUMNS_BY_NAME if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name}: missing named columns {missing}")
    return df.loc[:, list(SELECTED_COLUMNS_BY_NAME)]


def standardise_points(raw: pd.DataFrame) -> pd.DataFrame:
    vt = pd.to_numeric(raw["VesselType"], errors="coerce")
    lat = pd.to_numeric(raw["LAT"], errors="coerce")
    lon = pd.to_numeric(raw["LON"], errors="coerce")
    mmsi = pd.to_numeric(raw["MMSI"], errors="coerce")
    keep = (
        vt.between(CARGO_VESSELTYPE_MIN, CARGO_VESSELTYPE_MAX, inclusive="both")
        & lat.notna()
        & lon.notna()
        & (lat >= -90)
        & (lat <= 90)
        & (lon >= -180)
        & (lon <= 180)
        & mmsi.notna()
        & raw["portid"].astype("string").str.strip().ne("")
    )
    df = pd.DataFrame(
        {
            "portid": raw.loc[keep, "portid"].astype("string").str.strip(),
            "portname": raw.loc[keep, "portname"].astype("string"),
            "mmsi": mmsi.loc[keep].astype("int64"),
            "imo": normalize_imo(raw.loc[keep, "IMO"]),
            "vessel_name": raw.loc[keep, "VesselName"].astype("string"),
            "vessel_type": vt.loc[keep],
            "length": pd.to_numeric(raw.loc[keep, "Length"], errors="coerce"),
            "sog": pd.to_numeric(raw.loc[keep, "SOG"], errors="coerce"),
            "ts": pd.to_datetime(raw.loc[keep, "BaseDateTime"], utc=True, errors="coerce").dt.tz_localize(None),
            "lon": lon.loc[keep],
            "lat": lat.loc[keep],
        }
    )
    df = df.dropna(subset=["portid", "mmsi", "ts"])
    return df


def points_to_calls(points: pd.DataFrame) -> pd.DataFrame:
    """Vectorised equivalent of extract_port_calls_pilot.points_to_calls
    with frozen min_hours=3.0, sog_max=0.5, gap_hours=6.0.
    """
    if points.empty:
        return pd.DataFrame()
    df = points.sort_values(list(SORT_KEYS)).copy()
    df = df.loc[sog_eligible(df["sog"])].copy()
    if df.empty:
        return pd.DataFrame()
    dt = df.groupby(list(SEGMENT_ENTITY), sort=False)["ts"].diff()
    new_seg = dt.dt.total_seconds().div(3600.0).gt(GAP_HOURS).fillna(False)
    df["seg"] = new_seg.groupby([df["portid"], df["mmsi"]], sort=False).cumsum()
    g = df.groupby(["portid", "mmsi", "seg"], sort=False)
    imo_nonmiss = df["imo"].notna()
    nunique = (
        df.loc[imo_nonmiss]
        .groupby(["portid", "mmsi", "seg"], sort=False)["imo"]
        .nunique()
        .rename("nunique_imo_nonmissing")
    )
    imo_counts = (
        df.loc[imo_nonmiss]
        .groupby(["portid", "mmsi", "seg", "imo"], sort=False)
        .size()
        .rename("imo_n")
        .reset_index()
        .sort_values(["portid", "mmsi", "seg", "imo_n"], ascending=[True, True, True, False])
    )
    modal = imo_counts.drop_duplicates(["portid", "mmsi", "seg"])[
        ["portid", "mmsi", "seg", "imo"]
    ].rename(columns={"imo": "modal_imo"})
    out = g.agg(
        portname=("portname", "first"),
        imo=("imo", "first"),
        vessel_name=("vessel_name", "first"),
        vessel_type=("vessel_type", "first"),
        length=("length", "first"),
        arrival_ts=("ts", "min"),
        departure_ts=("ts", "max"),
        n_points=("ts", "size"),
        mean_sog=("sog", "mean"),
        mean_lon_lowspeed=("lon", "mean"),
        mean_lat_lowspeed=("lat", "mean"),
        n_imo_nonmissing=("imo", "count"),
    ).reset_index()
    out = out.merge(nunique.reset_index(), on=["portid", "mmsi", "seg"], how="left")
    out = out.merge(modal, on=["portid", "mmsi", "seg"], how="left")
    out["nunique_imo_nonmissing"] = out["nunique_imo_nonmissing"].fillna(0).astype("int64")
    out["port_hours"] = (out["departure_ts"] - out["arrival_ts"]).dt.total_seconds() / 3600.0
    out = out.loc[out["port_hours"] >= MIN_HOURS].copy()
    out = out.sort_values(["portid", "mmsi", "arrival_ts"]).drop(columns=["seg"])
    out["visit_number"] = out.groupby(["portid", "mmsi"]).cumcount() + 1
    out["first_eq_modal"] = (
        out["imo"].astype("string").fillna("") == out["modal_imo"].astype("string").fillna("")
    )
    return out


def attach_window_mean_lon(calls: pd.DataFrame, all_points: pd.DataFrame) -> pd.DataFrame:
    """Production LA/LB statistic: unweighted mean lon/lat of all filtered points
    in the call window (72_extract_wait_berth.py avg(lon)), not first-point lon.
    """
    if calls.empty:
        calls["mean_lon"] = pd.Series(dtype="float64")
        calls["mean_lat"] = pd.Series(dtype="float64")
        calls["n_points_window"] = pd.Series(dtype="int64")
        return calls
    import duckdb

    c = calls.reset_index(drop=True).copy()
    c["call_uid"] = np.arange(len(c), dtype=np.int64)
    pts = all_points[["portid", "mmsi", "ts", "lon", "lat"]].copy()
    con = duckdb.connect()
    con.register("calls", c[["call_uid", "portid", "mmsi", "arrival_ts", "departure_ts"]])
    con.register("pts", pts)
    stats = con.execute(
        """
        SELECT
          c.call_uid,
          avg(p.lon) AS mean_lon,
          avg(p.lat) AS mean_lat,
          count(*) AS n_points_window
        FROM calls c
        JOIN pts p
          ON p.portid = c.portid
         AND p.mmsi = c.mmsi
         AND p.ts >= c.arrival_ts
         AND p.ts <= c.departure_ts
        GROUP BY 1
        """
    ).fetchdf()
    con.close()
    out = c.merge(stats, on="call_uid", how="left").drop(columns=["call_uid"])
    return out


def assign_lalb(calls: pd.DataFrame) -> pd.DataFrame:
    df = calls.copy()
    sp = df["portid"].astype(str).eq(SAN_PEDRO_PORTID)
    has = sp & df["mean_lon"].notna()
    df["lalb_side"] = ""
    df.loc[has & (df["mean_lon"] < LALB_SPLIT_LON), "lalb_side"] = "Los_Angeles"
    df.loc[has & (df["mean_lon"] >= LALB_SPLIT_LON), "lalb_side"] = "Long_Beach"
    df["relationship_port"] = df["portname"].astype("string")
    df.loc[df["lalb_side"].eq("Los_Angeles"), "relationship_port"] = "Los_Angeles"
    df.loc[df["lalb_side"].eq("Long_Beach"), "relationship_port"] = "Long_Beach"
    return df


def stable_call_id(df: pd.DataFrame) -> pd.Series:
    arr = pd.to_datetime(df["arrival_ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    dep = pd.to_datetime(df["departure_ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return (
        df["mmsi"].astype("int64").astype(str)
        + "_"
        + df["portid"].astype(str)
        + "_"
        + arr
        + "_"
        + dep
    )
