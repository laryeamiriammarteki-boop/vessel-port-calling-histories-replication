from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    assert (ROOT / "README.md").exists()
    assert (ROOT / "LICENSE").exists()
    assert (ROOT / "run_all.py").exists()
    assert (ROOT / "config" / "paths_config.example.json").exists()


def test_no_local_paths_config_committed():
    assert not (ROOT / "config" / "paths_config.local.json").exists()
    assert not (ROOT / "paths_config.local.json").exists()


def test_aggregates_present():
    files = list((ROOT / "data" / "aggregates").glob("*.csv"))
    assert files, "expected aggregate CSVs"
    assert any(p.name.endswith("_REDACTED.csv") or p.name == "between_share.csv" for p in files)
