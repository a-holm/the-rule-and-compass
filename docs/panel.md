# Point-in-time panels, costs and the 2026H2 lockbox

Issue **#145**. The old crypto panel was selected with 2026 liquidity and
contained no delisted coins, so early training was a survivor set and the
long-only result was partly a survivor-drift bet. This page documents the
rebuild and the rules a runner must follow.

## Point-in-time membership

```sh
rcsr panel membership --data daily.csv --top 100 --lookback 90 \
  --min-history 90 --rebalance 30 --out membership.csv
```

At each rebalance (`as_of`, UTC midnight), candidates are ranked by trailing
average daily quote volume over `[as_of - lookback, as_of)` — **strictly
before** the decision instant. A symbol is eligible only if it is at least
`min-history` old and has `min-bars` bars in the window. The top `--top` names
are frozen until the next rebalance. No future bar, future volume or
survivor-only availability filter is read, so a coin that later delists was
still selectable while it traded.

The output is `as_of,symbol,member,adv_usd,n_bars`. Membership decisions (both
selected and not) are written so the builder can start and stop a symbol
cleanly; `member=0` is not a gap.

## Panel rebuild

```sh
rcsr panel build --data daily.csv --membership membership.csv \
  --out panel.rcsb [--bar-seconds 86400] [--migrations 'OLD->NEW:k[:tol]']
```

`build` keeps exactly the real bars of symbols that are members at that
instant:

* a delisted symbol stops at its last bar — there is no synthetic tail;
* a listing starts at its first bar — there is no pre-listing fill;
* a membership exclusion starts a new chain, so it is not reported as a data
  gap;
* no value is forward-filled, interpolated or carried across a listing,
  delisting or migration;
* `gap` counts missing `--bar-seconds` buckets inside a continued series
  (86400 daily, 3600 hourly, 900 for 15m);
* `member_asof` records which decision governs each row.

Migrations are applied before membership with the verified
[`ohlcv::splice`](ohlcv.md): the measured boundary ratio must be within
tolerance of the declared `k`, prices are divided by `k`, base volume is
multiplied by `k`, and the overlap is removed. A wrong `k` is a hard error.

## Resumable builds

A full hourly rebuild from 25,000 monthly archives is longer than one short
step, so `build` can cache work per symbol:

```sh
rcsr panel build --data hourly_base.rcsb --membership membership.csv \
  --bar-seconds 3600 --resume work/panel-parts --out panel.rcsb
```

Each symbol writes one atomic `<symbol>.part`; a re-run reads existing parts and
skips those symbols, printing `panel build: resume N/M symbols (P%), K part(s)
reused`. `state.txt` pins the bar step and membership row count, so a part built
from a different membership is refused rather than silently mixed in. The part
cache is deterministic: the one-shot and resumed builds have the same rows,
gaps and delisted set (a test pins this).

`rcsr prep` is **not** resumable; a full raw-archive rebuild re-parses every ZIP.
The measured stages are reported below.

## Audit and canaries

```sh
rcsr panel audit --data daily.csv --top 100 --lookback 90
```

`audit` prints per-year symbol counts, the delisted list, declared splice
canaries, undeclared level jumps and long gaps. On the historical research
panel it flags LUNA's 2022-09 level jump and its 138-day outage, so an
undeclared migration cannot enter level/EWM features silently. A canary fails
on TOMO's measured 1.0169 against `k=1`, while GAL's 59.9953 against `k=60`
passes.

## Costs and exits

```rust
let cost = CostModel { fee_bps: 10.0, half_spread_bps: 0.0, impact_bps: 0.0 };
let (net, ledger) = net_returns(&gross, &positions, &groups, &cost, true);
```

The label uses `cost.label_cost() = 2 * per_side_bps`; the evaluator charges
every position change and, with `charge_final_exit`, the last position of every
group. A one-round-trip fixture therefore agrees exactly between label and
evaluator. `buy_and_hold` pays the same entry and exit. A missing final exit
would flatter the strategy by one side (a test pins this).

The engine walk-forward also charges the final exit per group and refuses a
training or test window overlapping the lockbox.

## Matched-exposure benchmark

`matched_exposure_random_long(positions, groups, seed)` permutes the long/flat
positions **within each group**, preserving the exact number of long rows and
entries. A strategy that only "earns drift with a small denominator" cannot
beat this benchmark by construction. Report exposure, trades, turnover and beta
beside Sharpe.

## The 2026H2 lockbox

```text
[2026-07-01T00:00:00Z, 2027-01-01T00:00:00Z)
```

One guard, `rcsr::lockbox::Guard`, is used by every path that can train or
evaluate:

| runner | checked range |
|---|---|
| `rcsr run` (default search) | the actual train rows **and** the actual test rows after the split |
| `rcsr walkforward` | every window's `train_start..train_end` **and** `test_start..test_end` |
| `rcsr panel build --start/--end` | the explicit output range (a full build is provenance and is allowed) |
| `rcsr --cv` (cross-validation) | any table row in 2026H2, because folds are not the split |
| `rcsr prep --label` | labelled rows in the output |
| `rcsr panel audit` / `rcsr audit` | read-only: the `lockbox` coverage block is reported, never scored |

`--allow-lockbox` is the only override; it is for one declared final test, not
routine runs. `rcsr walkforward` echoes it in its JSON as
`"lockbox": {"permitted": true, "evaluated": true, ...}`. The check is skipped
when a time column is absent or is not Unix seconds (a row index or a year
fraction), so non-crypto data is not rejected by a crypto date.

## Provenance manifest

`rcsr panel build` writes `<out>.manifest.json` with input paths, SHA-256 and
byte sizes, the membership rebalance timestamps, decision/selected counts, panel
row and symbol counts, per-year symbol and row counts, the delisted list, data
gaps, the first/last dates and the output checksum.

It also writes a `lockbox` block, whether or not a row overlaps:

```json
"lockbox": {"start": 1782864000, "end": 1798761600,
            "start_date": "2026-07-01", "end_date": "2027-01-01",
            "permitted": false, "complete": false, "evaluated": false,
            "rows": 198567, "first_ts": 1782864000, "last_ts": 1798689599,
            "windows": [{"first_ts": ..., "last_ts": ..., "rows": ...}]}
```

`rows`/`windows` are counted from the retained panel rows, grouped by UTC month.
`complete` is false for a pre-2027 archive (Oct–Dec 2026 monthly files are not
published until 2027-02); `evaluated` says whether a declared final test ran.
`rcsr prep` writes the same block into its manifest, and `rcsr audit` includes it
in the JSON report.

`--snapshot <file>` embeds a verified archive-snapshot JSON verbatim as the
manifest's `snapshot` block, so vendor checksums, `coverage_last` and the
`complete` flag travel with the panel. The snapshot file is produced by the
fetch/verify step described below; the builder rejects anything that is not a
JSON object.

## Snapshot provenance (archive-only)

The public Binance Vision monthly archive was fetched and checked against the
vendor's sibling `.CHECKSUM` on 2026-10-21. All five samples matched:

| archive | bytes | vendor `Last-Modified` | SHA-256 (vendor = local) |
|---|---:|---|---|
| `BTCUSDT-15m-2026-09.zip` | 157,264 | 2026-10-05 11:13:54 | `0d452e88…48859cd0` |
| `ETHUSDT-15m-2026-09.zip` | 150,670 | 2026-10-05 11:36:47 | `88bb25ae…916b4b1f` |
| `LUNAUSDT-15m-2022-05.zip` | 59,359 | 2022-06-02 05:36:08 | `cfdd4a9a…924434c7` |
| `MATICUSDT-15m-2024-09.zip` | 38,523 | 2024-10-15 12:19:23 | `6fe10fce…e6c404c2` |
| `EOSUSDT-15m-2025-05.zip` | 108,222 | 2025-06-02 07:09:10 | `9ffbb0a0…67f70026` |

Monthly archives exist for 2026-07, 2026-08 and 2026-09 (HTTP 200) and are
**404 for 2026-10, 2026-11 and 2026-12**. The newest published monthly file is
therefore 2026-09, and a *complete* 2026H2 snapshot cannot exist before the
2027-02 monthly publication. A daily capture (`spot/daily/...`) is published
through 2026-10-20 and is a different source requiring reconciliation; it is not
used as a substitute for the monthly snapshot. Any snapshot built from the
archive is marked `coverage_last: 2026-09-30`, `complete: false`.

## Measured rebuild (2026-10)

Input: `pilot/panel_daily_all.csv` (563 symbols, 2018-01..2026-06; 2026H2
already excluded), `top=100`, `lookback=90`, `min-history=90`.

| panel | rows | symbols (union) | delisted | span | store | build |
|---|---|---|---|---|---|---|
| old `panel_daily_100` | 154,674 | 100 | 0 | 2021-05..2026-09 | 38 MB (local) | Python |
| new PIT daily | 232,006 | 432 | 129 | 2018-04..2026-06 | 23.2 MB | 16.9 s |
| old hourly compact | — | 100 | 0 | — | 1.8 GB `.rcsb` | Python |
| old hourly PIT (109-name input) | 3,515,245 | 108 | 19 | 2021-01..2026-09 | 336 MB | 11.2 s |
| **new PIT hourly (full archive)** | **4,775,757** | **445** | **332** | **2021-01..2026-06** | **478 MB** | **55.8 s** build |

The new hourly panel is built from `raw_15m_all` (636 symbol directories,
24,952 monthly ZIPs, 2017-08..2026-06), not the old 109-name survivor download:

```sh
rcsr prep --input raw_15m_all --format binance --resample 1h --jobs 8 \
  --out hourly_base.rcsb                       # 17,925,089 rows, 1.50 GB, 4m11s
rcsr panel membership --data hourly_base.rcsb --start 2021-01-01 \
  --end 2026-07-01 --top 100 --lookback 90 --min-history 90 --min-bars 60 \
  --rebalance 30 --out membership_hourly_pit.csv   # 67 rebalances, 9.7 s
rcsr panel build --data hourly_base.rcsb --membership membership_hourly_pit.csv \
  --bar-seconds 3600 --start 2021-01-01 --end 2026-07-01 \
  --resume work/panel-parts --snapshot snapshot.json \
  --out panel_hourly_pit_2021_2026h1.rcsb          # 55.8 s first, 40.3 s resumed
```

Per-year symbol counts (union over the year, because names rotate monthly):
2021: 185, 2022: 199, 2023: 204, 2024: 196, 2025: 200, 2026H1: 169. Rows:
872,550 / 867,258 / 869,955 / 871,194 / 862,335 / 432,465. 9,544 gap buckets
inside continued series. Store SHA-256
`6caa29791e05d07b8eb1ce5af820d030a9d8e97c7eb6ca35bb3f1c2587eb2682`.
The panel stops at 2026-06-30, so its manifest lockbox block has `rows: 0`.
The panel stage is now under the 60 s cap (55.8 s from cold parts, 40.3 s when
all 586 parts are reused, with a byte-identical store). The **full** raw-archive
rebuild is still 4m11s for `prep` + 9.7s for membership + 55.8s for the panel
(316s total). The `prep` stage re-parses all 24,952 ZIPs and is the remaining
non-resumable part.

Membership selects at most 100 names per period; the per-year union is larger
because names rotate. The delisted list includes FTT, LUNA, TOMOUSDT, MATIC and
other names absent from the old survivor panel. The new daily panel contains no
2026H2 rows.

## Limits

The builder reports missing bars; it does not quarantine a feature window after
a splice. Features are built by `rcsr prep` (#123) with
`window_gap_mask`/`label_gap_mask`, and a panel built from a migrated series
must be joined only on `member_asof` rows. The old-vs-new comparison above uses
the archived research files as inputs, not a live vendor download.
