# 01_acquire: spot files, checksum verified

One job: download spot files from the vendor archive and verify every checksum.

## Inputs

**This run:**
- `DATA`, the raw directory named by the dispatcher

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/python-pipeline-retirement.md`
- `../../CONTEXT.md`

Do NOT load: perpetual or futures data, any HTTP client inside the crate.

## Process

1. If `DATA` is not set, stop and say so.
2. Use the documented `curl` recipe against `https://data.binance.vision/`: spot only, never futures or perpetuals. Download each file with its sibling `.CHECKSUM`; skip files that exist; keep `.404` markers.
3. The crate has no HTTP client. Downloads are shell, outside `rcsr`.
4. Write `acquire.md`: `DATA`, file count, date range, symbols.

## Outputs

- `acquire.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test -n "$DATA" && (cd "$DATA" && sha256sum -c --quiet ./*.CHECKSUM) && test -z "$(find "$DATA" -ipath '*futures*' -o -ipath '*perp*' | head -1)"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
