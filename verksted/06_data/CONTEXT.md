# 06_data: from an unseen dataset to a verified, lock-checked store

One job: turn raw files of any supported data type into a verified `.rcsb` store with a dataset card, a declared locked test, a feature view that fits the columns the source has, a passed audit and a recorded manifest.

Stages with the same number are alternatives: exactly one type stage `04_*` runs per dataset, chosen by `stages/03_profile/output/profile.md`. Standard types and their slots: [data-types](references/data-types.md). This is the one place where the folder number is not a strict order.

## Inputs

**This run:**
- the dataset id `DS` (kebab-case), the raw directory `DATA`, the output store path `STORE`
- the local area `$VLOCAL` (see `../_shared/local.md`)

**Reference (every run):**
- `../_shared/rules.md`
- `../_shared/local.md`
- `references/data-types.md`
- `references/type-stage-contract.md`
- `references/time-rules.md`
- `../../docs/time-series.md`

Do NOT load: the ticket and long-run pipelines, earlier runs' `output/`, the references of a type stage that does not run, any file under a `*-sealed/` directory, any data file dated on or after the lock start.

## Process

1. [01_intake](stages/01_intake/CONTEXT.md): `stages/01_intake/` writes the dataset card: source, licence, earliest bar, lock window, question.
2. [02_acquire](stages/02_acquire/CONTEXT.md): `stages/02_acquire/` downloads and verifies checksums; files inside the lock are checksummed and never opened.
3. [03_profile](stages/03_profile/CONTEXT.md): `stages/03_profile/` profiles development files by name, header and count and names the type stage.
4. One type stage: [04_ohlcv](stages/04_ohlcv/CONTEXT.md) for OHLC and OHLCV bars, or [04_new_type](stages/04_new_type/CONTEXT.md) for a dataset no standard type covers.
5. [05_panel_pit](stages/05_panel_pit/CONTEXT.md): `stages/05_panel_pit/` builds point-in-time membership and the panel, or records "single series".
6. [06_enrich](stages/06_enrich/CONTEXT.md): `stages/06_enrich/` appends the feature view and the labels.
7. [07_audit_lock](stages/07_audit_lock/CONTEXT.md): `stages/07_audit_lock/` audits leakage and proves the lock holds.
8. [08_verify_manifest](stages/08_verify_manifest/CONTEXT.md): `stages/08_verify_manifest/` verifies hashes and writes the dataset manifest.
9. [09_lessons](stages/09_lessons/CONTEXT.md): `stages/09_lessons/` records what this dataset taught.

## Outputs

- `stages/08_verify_manifest/output/report.md`: store, hashes, audit verdict, deviations
- the dataset folder `datasets/<DS>/` in the local area: `card.md`, `adapt.md`, `manifest.md`

## Check

```sh
for s in 01_intake 02_acquire 03_profile 05_panel_pit 06_enrich 07_audit_lock 08_verify_manifest 09_lessons; do test "$(head -1 "$V/06_data/stages/$s/output/check.md")" = pass || exit 1; done; set -- "$V"/06_data/stages/04_*/output/check.md; test "$#" = 1 && test "$(head -1 "$1")" = pass
```
