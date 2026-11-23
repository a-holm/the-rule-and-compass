# 02_acquire: files downloaded and checksum verified

One job: download the source files and verify every checksum, without opening a file inside the lock.

## Inputs

**This run:**
- `../01_intake/output/card.md`
- `../01_intake/output/check.md` (must be `pass`)
- `DATA`, the acquisition root named by the dispatcher, and `IV`, the interval

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../references/time-rules.md`
- `../../../../docs/data-pipeline.md`
- `../../CONTEXT.md`

Do NOT load: perpetual or futures data, any HTTP client inside the crate. Build the tool once: `cargo build --release --manifest-path tools/acquire/Cargo.toml` (through the lock).

## Process

1. If `DATA` is not set, stop and say so.
2. `tools/acquire/target/release/rcsr-acquire fetch --symbols-file <list> --interval "$IV" --from <YYYY-MM> --to <YYYY-MM> --out "$DATA"`. Spot only, never futures or perpetuals. Existing zips are skipped; every request, including 404, is a row in `$DATA/downloads_$IV.csv`.
3. `rcsr-acquire verify --dir "$DATA" --interval "$IV" --snapshot "$DATA/snapshot_$IV.json"` checks each zip against its vendor `.CHECKSUM`. The crate has no HTTP client; `curl` is the transport of the tool.
4. Files dated at or after the lock start are fetched and checksum-verified only: fetch them with `--out "${DATA}-sealed"` and verify them there. The build never stages that directory, and no command lists their content.
5. Sources without a vendor checksum: write `sha256sum` of every file to `$DATA/manifest.sha256` at download time; the check verifies it.
6. Raw files never enter git.
7. Write `acquire.md`: `DATA`, file count, date range, symbols, and `licence:` copied from the card.

## Outputs

- `acquire.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
a="$V/06_data/stages/02_acquire/output/acquire.md"; d=$(sed -n 's/^dev_end: *\([0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}\).*/\1/p' "$V/06_data/stages/01_intake/output/card.md"); test -n "$DATA" && test -n "$d" && grep -qE '^licence: .+' "$a" && if test -f "$DATA/manifest.sha256"; then (cd "$DATA" && sha256sum -c --quiet manifest.sha256); else tools/acquire/target/release/rcsr-acquire verify --dir "$DATA" --interval "$IV" --snapshot "$DATA/snapshot_$IV.json"; fi && test -z "$(find "$DATA" -type f -printf '%f\n' | awk -v d="$d" 'match($0,/[0-9][0-9][0-9][0-9]-[0-9][0-9](-[0-9][0-9])?/){s=substr($0,RSTART,RLENGTH); if(s>=substr(d,1,RLENGTH)) print}' | head -1)" && test -z "$(find "$DATA" -ipath '*futures*' -o -ipath '*perp*' | head -1)"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
