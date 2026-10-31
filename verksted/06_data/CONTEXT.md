# 06_data: from raw market files to a verified, leak-checked store

One job: turn raw spot market files into a verified `.rcsb` store with point-in-time membership, appended features, a passed audit and a verified manifest.

Evidence: the crypto work of 2026-10-19 to 2026-10-22 and the retirement of 41 Python scripts (`../../docs/python-pipeline-retirement.md`) into `rcsr prep`, `rcsr panel`, `rcsr enrich`, `rcsr audit` and `rcsr verify-manifest`. Acquisition is shell today; a Rust tool is a separate issue and replaces stage 01 only.

## Inputs

**This run:**
- the raw data directory `DATA`, the output store path `STORE` and the target column

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/output-format.md`
- `../../docs/panel.md`
- `../../docs/ohlcv.md`
- `../../docs/data-format.md`
- `../../docs/time-series.md`
- `../../docs/leakage.md`

Do NOT load: the ticket and long-run pipelines, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. `stages/01_acquire/`: spot files with their vendor checksums.
2. `stages/02_import_prep/`: `rcsr import` or `rcsr prep` into a store with a manifest.
3. `stages/03_panel_pit/`: point-in-time membership and panel, delisted coins kept.
4. `stages/04_enrich/`: append features, prove CPU and GPU agree.
5. `stages/05_audit_lockbox/`: leakage audit and the 2026H2 lockbox refusal.
6. `stages/06_verify_manifest/`: verify the manifest against the data, write the report.

## Outputs

- `stages/06_verify_manifest/output/report.md`: store, hashes, audit verdict, deviations
- `stages/06_verify_manifest/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/06_data/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
