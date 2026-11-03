# Vessel–port calling histories: replication package

**Status:** Local public-package root (privacy-safe). Not yet pushed to a remote.

This package reproduces selected analytical checks for a study of vessel–port calling histories constructed from public U.S. AIS broadcast points: dwell/turnaround measures, history associations, forecast/scheduling sensitivity, and port comparison aggregates.

It is a **code and aggregate-data replication package**, not a manuscript archive.

## What is included

- Portable Python scripts under `code/`
- Path-redacted aggregate frozen result tables under `data/aggregates/`
- Redacted AIS acquisition recipe under `data/source_references/`
- Technical docs under `docs/`
- Example path config: `config/paths_config.example.json`
- Software pins: `requirements.txt`, `environment.yml`

## What is not included

- Nationwide NOAA AIS day files (obtain from the public source)
- Vessel-level analysis panels (restricted / author-gated)
- Manuscripts, cover letters, or author-identifying materials

## Data access

1. Public AIS source: [NOAA Marine Cadastre Vessel Traffic (AIS)](https://marinecadastre.gov/ais/)  
   Study coverage years: **2021–2025** (see redacted source manifest).
2. Copy `config/paths_config.example.json` → `config/paths_config.local.json` and set local paths (keep `*.local.json` gitignored).
3. Aggregate tables in `data/aggregates/` support correspondence checks against archived headline outputs.

## Quick start (PowerShell)

```powershell
cd "PATH\TO\TP_PUBLIC_REPLICATION_PACKAGE"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy config\paths_config.example.json config\paths_config.local.json
# Edit paths_config.local.json for your machine, then:
python run_all.py --smoke
```

Full reproduction of estimation stages requires local derived panels that are **not** redistributed here. See `docs/REPRODUCE.md`.

## Layout

| Path | Contents |
|------|----------|
| `code/data_construction/` | Call-construction portable script |
| `code/statistical_estimation/` | Stage-2 portable runner + pilot schema helper |
| `code/forecast_scheduling/` | C4B sensitivity + BTS Spearman check |
| `data/aggregates/` | Path-redacted frozen aggregate CSVs |
| `data/source_references/` | Redacted AIS year/file recipe |
| `docs/` | Data dictionary, availability notes, reproduce notes |
| `tests/` | Smoke / structural tests |
| `config/` | Example path configuration |

## Software

See `requirements.txt` / `environment.yml`. Historically verified stack notes are summarized in `docs/SOFTWARE.md`.

## Licence

MIT — see `LICENSE` (copyright: Study Authors).

## Third-party data

- NOAA Marine Cadastre AIS: cite the public source; do not claim ownership; do not re-host nationwide extracts in this repository.
- Any BTS validation excerpt: confirm redistribution terms before broader reuse.
