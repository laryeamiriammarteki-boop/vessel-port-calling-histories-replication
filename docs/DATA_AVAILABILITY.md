# Data availability (package-level)

## Public source AIS

Raw AIS broadcast points: NOAA Marine Cadastre Vessel Traffic (AIS)  
https://marinecadastre.gov/ais/  

This package does **not** redistribute nationwide AIS day files. Reconstruct extracts using `data/source_references/02_SOURCE_MANIFEST_REDACTED.csv`.

## Included here

Path-redacted aggregate frozen analytical tables under `data/aggregates/` and portable scripts under `code/`.

## Restricted

Vessel-level panels with vessel identifiers and relationship-specific calling histories are **not** included. They remain author-gated because of privacy and re-identification risk in a small multi-port network.

## Repository URL / DOI

- Blinded review mirror of this package: https://anonymous.4open.science/r/vessel-port-calling-histories-replication-8314/
- An open (non-blinded) code repository also hosts the same privacy-safe package; its identifying URL is omitted here for anonymous peer review and can be supplied after acceptance.
- Research-data DOI: none issued
- License: MIT (Study Authors)
