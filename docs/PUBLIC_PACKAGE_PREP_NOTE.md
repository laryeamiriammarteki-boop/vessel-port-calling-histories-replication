# Public package preparation note

**Rule installed:** `C:\Users\WIT_User\Documents\Vessel call a\.cursor\rules\public-github-and-anonymous-4open.mdc`

**Public package root:** `F:\DATA\TP_VESSEL_PORT_CALLING_HISTORIES_REPLICATION\TP_PUBLIC_REPLICATION_PACKAGE`

**Actions completed:**
- Copied portable scripts only (excluded `*_ORIGINAL_SNAPSHOT.py` with absolute user paths)
- Copied redacted aggregate CSVs + redacted AIS source manifest
- Added privacy-safe README, LICENSE (Study Authors), docs, `run_all.py --smoke`, tests, `.gitignore`
- Local `git init` + initial commit only
- **No GitHub remote, no push, no anonymous.4open deposit**

**Next (when authorized):**
1. `gh repo create ... --public --source . --remote origin --push` from the public package root
2. Hard-refresh GitHub; confirm no manuscript/identity files
3. Mirror on https://anonymous.4open.science/ for blinded review
4. Fill Draft B data-availability placeholders with the verified URLs only
