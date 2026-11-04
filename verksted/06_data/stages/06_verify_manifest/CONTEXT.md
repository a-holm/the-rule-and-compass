# 06_verify_manifest: hashes match the data

One job: verify the manifest against the data and write the report.

## Inputs

**This run:**
- `../05_audit_lockbox/output/audit.md`
- `../05_audit_lockbox/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: re-running earlier stages.

## Process

1. If an input is missing or empty, stop and say which.
2. `rcsr verify-manifest "$ENRICHED.manifest.json" --data "$ENRICHED"`.
3. Write `report.md` per `output-format.md`: store paths, hashes, audit verdict, the survivorship statement, deviations.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
"$RUN/my-version/target/release/rcsr" verify-manifest "$ENRICHED.manifest.json" --data "$ENRICHED"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
