# Walk-forward benchmark protocol

Issue #124 asks that `rcsr walkforward` evaluates a 4M-row panel in seconds
while reading the `.rcsb` store directly. This document pins the corpus, the
window, the command and the check, so a recorded number always refers to the
same fixture.

The pilot corpus is git-ignored and lives in the main checkout, so a linked
worktree must point at it explicitly. Run from the repository root:

```sh
/usr/bin/time -f '%e s wall, %M KiB peak RSS' \
  my-version/target/release/rcsr walkforward \
  --data arbeidsmappe/crypto-rcsr/pilot/panel_hourly_v2.rcsb --target y60 \
  --expr r16 --vars r16 --time ts --group-by symbol --horizon 172800 \
  --window '2024,1609459200,1704067200,1704240000,1735689600' \
  --cost-bps 10 --threshold-quantile 0.7 --periods-per-year 8760 --json
```

The first run warms the page
cache. The recorded wall time is the median of two warm runs.

## Corpus

| item | value |
|---|---|
| store | `arbeidsmappe/crypto-rcsr/pilot/panel_hourly_v2.rcsb` |
| store SHA-256 | `7b98a2836f97b4d6ac6385443486a81f73fec929de310f28d6a4efa3cb78a6f4` |
| rows x columns | 3,943,856 x 56 |
| source CSV | `arbeidsmappe/crypto-rcsr/pilot/panel_hourly_compact.csv`, 4,233,483,400 bytes, SHA-256 `7a3233822af84dbe68f73f005627713fdc74f4ad31e0e4db023f6dd047ed5bca` |

Use the **v2** store: it is self-describing (`ts` as `i64`, `symbol` as `u32`).
If it is absent, rebuild it from the source CSV with
`rcsr import arbeidsmappe/crypto-rcsr/pilot/panel_hourly_compact.csv
arbeidsmappe/crypto-rcsr/pilot/panel_hourly_v2.rcsb --group-rows 65536`. The
data is git-ignored pilot data, not a fixture to commit; the hashes above are
the pin.

## Window and arithmetic

| field | value |
|---|---|
| train | `[1609459200, 1704067200)`: 2021-01-01 to 2024-01-01 |
| test | `[1704240000, 1735689600)`: 2024-01-03 to 2025-01-01 |
| purge | 172,800 s (the `y60` horizon) |
| score | `r16` |
| target | `y60` |
| group | `symbol` |
| position | `1` if `score > q70(train scores)` and finite, else `0` (long/flat only) |
| quantile | q70, linear interpolation (type 7) |
| cost | 10 bp per side on `abs(position - previous position)` inside each symbol |
| aggregation | equal-weight mean net return per timestamp |
| annualisation | 8,760 hourly periods per year |

Every symbol starts flat. The evaluator charges each symbol's final exit, as
documented in `walkforward.md`. It drops rows whose score or target is
non-finite.

## Result (measured)

Intel i7-3770K, 31 GiB, Linux, release binary, warm cache, median of two runs:

| command | wall | peak RSS | q70 threshold | test rows |
|---|---:|---:|---:|---:|
| `rcsr walkforward --group-by symbol --horizon 172800` | 4.06 s | 107.7 MiB | `0.007034873088967409` | 845,922 |

Absolute seconds depend on the host and the page cache. Compare a new
measurement against this row on the same host.
