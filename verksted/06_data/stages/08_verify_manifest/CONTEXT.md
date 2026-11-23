# 08_verify_manifest: hashes match the data

One job: verify the store hashes, write the dataset manifest and file the dataset folder in the local area.

## Inputs

**This run:**
- `../07_audit_lock/output/audit.md`
- `../07_audit_lock/output/check.md` (must be `pass`)
- `../01_intake/output/card.md`
- the `adapt.md` of the type stage that ran

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../../_shared/local.md`
- `../../CONTEXT.md`

Do NOT load: re-running earlier stages, raw data files, any value from inside the locked range.

## Process

1. If an input is missing or empty, stop and say which. The recipe is `../04_*/output/adapt.md`.
2. Verify the hashes. Single series: the `sha256sum` of `$ENRICHED` equals `output_sha256` of `$ENRICHED.enrich.json`. Panel: `rcsr panel schema --data "$PANEL_ENRICHED" --verify --lock-record <record>` exits 0 (exit 3 is a mismatch). In both cases the `sha256sum` of `$STORE` equals `output_sha256` of `$STORE.manifest.json`. `rcsr verify-manifest` is not used here: it reads run manifests only and refuses a store manifest.
3. Write `manifest.md`: store paths, sha256 of every store and sidecar that matters, row counts, first and last development bar, binary sha256 and the commit it was built from, registry version, relist and migration lists, universe and exclusion list hashes, floors, zero-volume policy, deviations. No value from inside the lock, no raw row.
4. Write `report.md` per `output-format.md`: store paths, hashes, the audit verdict, the survivorship statement, deviations.
5. File the dataset: `verksted/_tools/local-init.sh --dataset "$DS" --card ../01_intake/output/card.md --adapt <adapt.md of the type stage> --manifest output/manifest.md` copies the three files to `datasets/<DS>/` of the local area.

## Outputs

- `report.md` -> `output/`
- `manifest.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
o="$V/06_data/stages/08_verify_manifest/output"; a=$(ls "$V"/06_data/stages/04_*/output/adapt.md); R="$(verksted/_tools/crate-dir.sh "$RUN")/target/release/rcsr"; test -s "$o/report.md" && test -s "$o/manifest.md" && for f in card adapt manifest; do test -s "${VLOCAL:?}/datasets/$DS/$f.md" || exit 1; done; test "$(sha256sum "$STORE" | cut -d' ' -f1)" = "$(jq -r .output_sha256 "$STORE.manifest.json")" && if grep -qx 'series: single' "$a"; then test "$(sha256sum "$ENRICHED" | cut -d' ' -f1)" = "$(jq -r .output_sha256 "$ENRICHED.enrich.json")"; else "$R" panel schema --data "$PANEL_ENRICHED" --verify; fi
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
