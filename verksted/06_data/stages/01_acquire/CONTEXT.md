# 01_acquire: spot files, checksum verified

One job: download spot files from the vendor archive and verify every checksum.

## Inputs

**This run:**
- `DATA`, the acquisition root named by the dispatcher, and `IV`, the interval

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/python-pipeline-retirement.md`
- `../../CONTEXT.md`

Do NOT load: perpetual or futures data, any HTTP client inside the crate. Build the tool once: `cargo build --release --manifest-path tools/acquire/Cargo.toml` (through the lock).

## Process

1. If `DATA` is not set, stop and say so.
2. `tools/acquire/target/release/rcsr-acquire fetch --symbols-file <list> --interval "$IV" --from <YYYY-MM> --to <YYYY-MM> --out "$DATA"`. Spot only, never futures or perpetuals. Existing zips are skipped; every request, including 404, is a row in `$DATA/downloads_$IV.csv`.
3. `rcsr-acquire verify --dir "$DATA" --interval "$IV" --snapshot "$DATA/snapshot_$IV.json"` checks each zip against its vendor `.CHECKSUM`. The crate has no HTTP client; `curl` is the transport of the tool.
4. Write `acquire.md`: `DATA`, file count, date range, symbols.

## Outputs

- `acquire.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test -n "$DATA" && tools/acquire/target/release/rcsr-acquire verify --dir "$DATA" --interval "$IV" --snapshot "$DATA/snapshot_$IV.json" && test -z "$(find "$DATA" -ipath '*futures*' -o -ipath '*perp*' | head -1)"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
