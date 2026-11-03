# Data availability (package-level)

## Public source AIS

Raw AIS broadcast points: NOAA Marine Cadastre Vessel Traffic (AIS)  
https://marinecadastre.gov/ais/  

This package does **not** redistribute nationwide AIS day files. Reconstruct extracts using `data/source_references/02_SOURCE_MANIFEST_REDACTED.csv`.

## Included here

Path-redacted aggregate frozen analytical tables under `data/aggregates/` and portable scripts under `code/`.

## Restricted

Vessel-level panels with vessel identifiers and relationship-specific calling histories are **not** included. They remain author-gated because of privacy / re-identification risk in a small multi-port network.

## Repository URL / DOI

- Public repository: https://github.com/laryeamiriammarteki-boop/vessel-port-calling-histories-replication
- Blinded review mirror: https://anonymous.4open.science/r/vessel-port-calling-histories-replication-8314/
- Research-data DOI: none issued
- License: MIT (Study Authors)
