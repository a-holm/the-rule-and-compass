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
test window overlapping the lockbox.

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

`rcsr walkforward` refuses every test window that intersects this range unless
`--allow-lockbox` is passed; that flag is for one declared final test, not for
routine runs. `rcsr panel build --start/--end` applies the same check when an
explicit evaluated range is given (a full panel may contain the lockbox; a
runner may not read it).

## Provenance manifest

`rcsr panel build` writes `<out>.manifest.json` with input paths, SHA-256 and
byte sizes, the membership rebalance timestamps, decision/selected counts, panel
row and symbol counts, per-year symbol and row counts, the delisted list, data
gaps, the first/last dates and the output checksum.

## Measured rebuild (2026-10)

Input: `pilot/panel_daily_all.csv` (563 symbols, 2018-01..2026-06; 2026H2
already excluded), `top=100`, `lookback=90`, `min-history=90`.

| panel | rows | symbols (union) | delisted | span | store | build |
|---|---|---|---|---|---|---|
| old `panel_daily_100` | 154,674 | 100 | 0 | 2021-05..2026-09 | 38 MB (local) | Python |
| new PIT daily | 232,006 | 432 | 129 | 2018-04..2026-06 | 23.2 MB | 16.9 s |
| old hourly compact | — | 100 | 0 | — | 1.8 GB `.rcsb` | Python |
| new PIT hourly | 3,515,245 | 108 | 19 | 2021-01..2026-09 | 336 MB | 11.2 s |

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
