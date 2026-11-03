# -*- coding: utf-8 -*-
"""PORTABLE COPY of reproduce_bts_spearman.py

Scientific method unchanged: frozen BTS extract + Stage 2 frame monthly
median port_hours Spearman on Jan 2022–Jun 2023 overlap.

Path change only: roots loaded from paths_config JSON (env TP_PATHS_CONFIG
or sibling/ancestor paths_config.local.json). Outputs write to PILOT_OUT.
Does not overwrite study frozen outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from scipy import stats

EXPECTED_FRAME_SHA = "9bcbf4b78ae022bd3b71c0077a1373c98996d3acf35df7a535d7855fbeaff9df"


def load_config() -> dict:
    env = os.environ.get("TP_PATHS_CONFIG")
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parent / "paths_config.local.json",
            here.parents[2] / "paths_config.local.json",
            Path(r"F:/DATA/TP_VESSEL_PORT_CALLING_HISTORIES_REPLICATION/local_repo_staging/paths_config.local.json"),
        ]
    )
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise SystemExit("paths_config.local.json not found; set TP_PATHS_CONFIG")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    frame_path = Path(cfg["STAGE2_FRAME"])
    bts_path = Path(cfg["BTS_CSV"])
    out_dir = Path(cfg["PILOT_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "COMMENT4_BTS_18MONTH_REPRODUCIBILITY_PILOT.csv"

    expected = cfg.get("EXPECTED_STAGE2_SHA256", EXPECTED_FRAME_SHA)
    frame_sha = sha256(frame_path)
    if frame_sha != expected:
        raise SystemExit(f"Stage 2 frame hash mismatch: {frame_sha}")

    bts = pd.read_csv(bts_path)
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    }
    bts["month_num"] = bts["Month"].map(month_map)
    bts["year_num"] = pd.to_numeric(bts["Year"], errors="coerce")
    bts["hours"] = pd.to_numeric(bts["Hours"], errors="coerce")
    bts = bts.dropna(subset=["month_num", "year_num", "hours"]).copy()
    bts["ym"] = [f"{int(y)}-{int(m):02d}" for y, m in zip(bts["year_num"], bts["month_num"])]

    df = pd.read_parquet(frame_path)
    expected_n = int(cfg.get("EXPECTED_N", 54788))
    expected_pairs = int(cfg.get("EXPECTED_PAIRS", 14022))
    if len(df) != expected_n or df["pair_id"].nunique() != expected_pairs:
        raise SystemExit("estimation-frame size mismatch")
    df = df.copy()
    df["port_hours"] = pd.to_numeric(df["port_hours"], errors="coerce")
    df["H_W12"] = pd.to_numeric(df["H_W12"], errors="coerce")
    df = df.dropna(subset=["port_hours", "H_W12", "pair_id", "relationship_port"]).copy()
    arr = pd.to_datetime(df["arrival_ts"], utc=True, errors="coerce")
    df["ym"] = arr.dt.tz_localize(None).dt.to_period("M").astype(str)
    ais_m = df.groupby("ym")["port_hours"].median().rename("ais_study_median_port_hours")
    ais_n = df.groupby("ym").size().rename("ais_n_calls")

    merged = bts.merge(ais_m.reset_index(), on="ym", how="left").merge(ais_n.reset_index(), on="ym", how="left")
    ov = merged.dropna(subset=["ais_study_median_port_hours", "hours"]).copy()
    ov = ov.sort_values("ym")
    if list(ov["ym"]) != [f"{y}-{m:02d}" for y in (2022, 2023) for m in range(1, 13) if not (y == 2023 and m > 6)]:
        raise SystemExit("overlap months are not January 2022-June 2023")

    rho, p = stats.spearmanr(ov["hours"], ov["ais_study_median_port_hours"])
    ov["bts_hours_as_stored"] = ov["hours"]
    ov["bts_value_is_integer"] = ov["hours"].map(lambda x: float(x).is_integer())
    ov["calendar_month_key_aligned"] = True
    ov["spearman_rho_unrounded"] = float(rho)
    ov["spearman_p_unrounded"] = float(p)
    ov["ais_population"] = "stage2_estimation_frame_n_54788_arrival_month_median"
    ov["bts_population"] = "nfsh-p62e_national_monthly_average_integer_hours"
    cols = [
        "ym",
        "bts_hours_as_stored",
        "ais_study_median_port_hours",
        "ais_n_calls",
        "bts_value_is_integer",
        "calendar_month_key_aligned",
        "spearman_rho_unrounded",
        "spearman_p_unrounded",
        "ais_population",
        "bts_population",
    ]
    ov[cols].to_csv(out, index=False)
    print(f"rows={len(ov)}")
    print(f"spearman={float(rho)!r}")
    print(f"p={float(p)!r}")
    print(f"ais_n_sum={int(ov['ais_n_calls'].sum())}")
    print(f"frame_sha={frame_sha}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
