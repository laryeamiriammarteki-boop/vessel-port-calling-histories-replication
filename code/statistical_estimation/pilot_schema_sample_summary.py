# -*- coding: utf-8 -*-
"""Bounded pilot step A/B: schema + deterministic sample summary on frozen STAGE2 frame."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import pandas as pd

def load_config():
    env = os.environ.get("TP_PATHS_CONFIG")
    cands = []
    if env: cands.append(Path(env))
    cands.append(Path(r"F:/DATA/TP_VESSEL_PORT_CALLING_HISTORIES_REPLICATION/local_repo_staging/paths_config.local.json"))
    for p in cands:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), p
    raise SystemExit("config missing")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.perf_counter()
    cfg, cfg_path = load_config()
    frame = Path(cfg["STAGE2_FRAME"])
    out = Path(cfg["PILOT_OUT"]); out.mkdir(parents=True, exist_ok=True)
    sha = sha256(frame)
    df = pd.read_parquet(frame)
    summary = {
        "config_path": str(cfg_path),
        "frame_path": str(frame),
        "frame_sha256": sha,
        "expected_sha256": cfg.get("EXPECTED_STAGE2_SHA256"),
        "sha_match": sha == cfg.get("EXPECTED_STAGE2_SHA256"),
        "n_rows": int(len(df)),
        "n_pairs": int(df["pair_id"].nunique()),
        "expected_n": cfg.get("EXPECTED_N"),
        "expected_pairs": cfg.get("EXPECTED_PAIRS"),
        "n_match": int(len(df)) == int(cfg.get("EXPECTED_N", -1)),
        "pairs_match": int(df["pair_id"].nunique()) == int(cfg.get("EXPECTED_PAIRS", -1)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "port_hours_median": float(pd.to_numeric(df["port_hours"], errors="coerce").median()),
        "port_hours_mean": float(pd.to_numeric(df["port_hours"], errors="coerce").mean()),
        "H_W12_median": float(pd.to_numeric(df["H_W12"], errors="coerce").median()),
        "relationship_port_nunique": int(df["relationship_port"].nunique()),
        "year_value_counts": {str(k): int(v) for k, v in df["year"].value_counts().sort_index().items()},
        "elapsed_sec": time.perf_counter() - t0,
    }
    (out / "PILOT_SCHEMA_AND_SAMPLE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    schema = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    schema.to_csv(out / "PILOT_STAGE2_SCHEMA.csv", index=False)
    print(json.dumps({k: summary[k] for k in ["sha_match","n_rows","n_pairs","n_match","pairs_match","port_hours_median","elapsed_sec"]}, indent=2))

if __name__ == "__main__":
    main()
