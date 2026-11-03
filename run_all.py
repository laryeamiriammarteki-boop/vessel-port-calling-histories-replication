#!/usr/bin/env python
"""Entry point for smoke checks on the public replication package."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def smoke() -> int:
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "requirements.txt",
        ROOT / "config" / "paths_config.example.json",
        ROOT / "data" / "aggregates" / "between_share.csv",
        ROOT / "data" / "source_references" / "02_SOURCE_MANIFEST_REDACTED.csv",
        ROOT / "code" / "statistical_estimation" / "run_stage2_portable.py",
        ROOT / "code" / "forecast_scheduling" / "reproduce_bts_spearman_portable.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        print("SMOKE FAIL — missing:", *missing, sep="\n  ")
        return 1
    # Structural: aggregates are non-empty CSVs
    for p in (ROOT / "data" / "aggregates").glob("*.csv"):
        if p.stat().st_size < 10:
            print("SMOKE FAIL — empty aggregate:", p.name)
            return 1
    print("SMOKE PASS — required files present; aggregates non-empty.")
    print("Full estimation still requires local AIS + gated derived panels.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run structural smoke checks only")
    args = parser.parse_args(argv)
    if args.smoke:
        return smoke()
    print("No action specified. Try: python run_all.py --smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
