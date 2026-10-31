# 02_import_prep: store with a manifest

One job: build the store from the raw files.

## Inputs

**This run:**
- `../01_acquire/output/acquire.md`
- `../01_acquire/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/data-format.md`
- `../../../../docs/ohlcv.md`
- `../../CONTEXT.md`

Do NOT load: panel membership, features.

## Process

1. If an input is missing or empty, stop and say which.
2. One CSV: `rcsr import in.csv "$STORE" --group-rows N`. A directory of raw files: `rcsr prep --input "$DATA" --out "$STORE" --resample 1h`.
3. Byte contract: UTC seconds as i64, NaN is a gap, zero-volume buckets stay with `traded=false`. Watch the 2025 ms to us switch of the time unit.
4. Write `prep.md`: command, exit code, row and symbol counts from the manifest.

## Outputs

- `prep.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test -s "$STORE" && test -s "$STORE.manifest.json"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
