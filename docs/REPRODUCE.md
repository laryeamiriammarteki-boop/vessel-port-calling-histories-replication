# Reproduce

## Distinction

1. **Archived-output correspondence** — compare script outputs to `data/aggregates/` (SHA-256 / numeric tolerances).
2. **Clean-environment full pipeline** — requires local AIS extracts plus author-gated derived panels that are not in this public package.

## Proposed order

1. Obtain NOAA AIS extracts per the redacted source manifest.
2. Configure `config/paths_config.local.json` from the example file.
3. Run construction → estimation → forecast/scheduling portable scripts as available.
4. Compare key aggregates to files in `data/aggregates/`.
5. Optional smoke entry: `python run_all.py --smoke`

## Hard stops

- Do not overwrite frozen aggregates to chase new runs.
- Do not commit `paths_config.local.json` or absolute machine paths.
- Do not publish vessel-level panels without separate clearance.
