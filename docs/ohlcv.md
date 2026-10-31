# OHLCV market-data layer

`rcsr::ohlcv` and `rcsr::features` (issue #125) give market data a typed shape
and make the classic look-ahead traps explicit: partial bars, sessions, DST,
multi-timeframe joins and corporate actions.

## Types

- `Bar { time, open, high, low, close, volume, quote_volume, trades,
  taker_buy_base, taker_buy_quote, complete, traded }`. `time` is the bucket
  start in seconds since the Unix epoch (UTC). `vwap()` is
  `quote_volume / volume` when base volume is positive.
- `Timeframe`: `1m 3m 5m 15m 30m 1h 2h 4h 1d 1w`. Intraday buckets are
  epoch-aligned UTC buckets; `1d`/`1w` are calendar-aware when resampled with a
  `Calendar`.

## Resampling

`resample(&bars, tf)` requires strictly increasing input and aggregates
`open=first`, `high=max`, `low=min`, `close=last`, and sums for volume, quote
volume, trades and taker-buy. `Resampler` is the streaming form: it holds one
bucket, not the series, and seals a bar when the bucket changes. Empty buckets
are **not** created: gaps remain gaps. A zero-volume bucket is kept and flagged
`traded=false`; dropping it would bias the sample. No forward fill ever happens.

`resample_as_of(&bars, tf, now)` keeps the bucket containing `now` but marks it
`complete=false`, and emits nothing after it. A live strategy must treat that
bar as tentative.

The fold is associative: `fold(1m -> 15m) -> 1h` equals `fold(1m -> 1h)`, and
the streaming and batch forms return the same bars.

## Calendars and sessions

`Calendar::Crypto247`, `Calendar::Nyse` and `Calendar::Lse`.

- Crypto 24/7: every UTC day is a session; no holidays, no DST.
- NYSE/Nasdaq RTH: 09:30-16:00 America/New_York, with the ten US market
  holidays (Saturday -> Friday, Sunday -> Monday), Good Friday, MLK, Presidents,
  Juneteenth, Memorial, Independence, Labor, Thanksgiving and Christmas.
- Half-days: the day after Thanksgiving, July 3 and December 24 (weekdays)
  close at 13:00 New York. LSE half-day is December 24.
- LSE RTH: 08:00-16:30 Europe/London with UK bank holidays.

Timestamps are UTC seconds internally; `timezone()` names the display zone.
DST is computed from the US/UK rules, so a 14:30 UTC open in January becomes a
13:30 UTC open after the March transition. `session_at`, `is_open`,
`next_session`, `session_on` and `bucket_start` are the public seam.

`resample_calendar` keys `1d`/`1w` buckets by the session open and refuses a base
bar outside a session, so a bar cannot silently cross a session boundary. A
crypto daily bucket is still the UTC day.

## Causal multi-timeframe joins

`join_causal(base, higher, higher_tf)` gives every base bar the last higher bar
whose bucket has **closed** at or before the decision instant:

```text
visible <=> higher.time + higher_tf.seconds() <= base.time
```

So a 15m decision at 10:00 sees the 09:00 hourly bar, not the 10:00 one, and
sees the previous day's daily bar, never the current day's close. The join
refuses a higher series that is not aligned to its timeframe (an unaligned bar
has no well-defined close instant). `join_causal_calendar` uses the calendar
close instead, and `join_column_causal` joins a scalar feature column.

`lookahead_violations(base, joined, higher_tf)` reports rows whose candidate
bar would not have closed; a correct join returns an empty vector, and a
deliberately future-shifted column is caught by the test suite.

### The 36-column multi-timeframe family (`rcsr enrich --features mtf`)

`src/enrich.rs` materialises twelve leaves for each of `1h`, `4h` and `1d`,
named `mtf_<timeframe>_<leaf>` (36 columns):

| leaf | definition, in higher-timeframe bars |
|---|---|
| `ret1`, `ret4`, `ret24` | `close_t/close_{t-k} - 1` |
| `ma_ratio` | `close_t/MA20_t - 1` |
| `vol` | population sd of one-bar simple returns over 20 bars |
| `atr_pct` | Wilder `ATR14_t/close_t` |
| `rsi` | `RSI14_t/50 - 1` |
| `close_pos` | `(close_t-low_t)/(high_t-low_t) - 0.5` |
| `donchian_pos` | `(close_t-min low20)/(max high20-min low20) - 0.5` |
| `range_pct` | `(high_t-low_t)/close_t` |
| `trend_slope` | `MA20_t/MA20_{t-20} - 1` |
| `bar_age` | seconds since the higher bar closed at the base decision time |

The base series is resampled per symbol with [`resample`] (empty buckets remain
gaps, no forward fill), the leaf is evaluated on the higher bars, and the
result is joined at the **last closed** higher bucket:

```text
visible <=> higher.time + higher_tf.seconds() <= base.time
```

A partial or current higher bucket is never visible, so a 10:15 decision on a
15 m base sees the 09:00 hourly bar (closed at 10:00), not the in-progress
10:00 bar. `bar_age` is exactly `base.time - (higher.time +
higher_tf.seconds())`: zero at the close instant and, for example, 900/1800/2700
seconds mid-hour on a 15 m base. Windows count higher-timeframe bars and require
contiguous higher buckets; a gap is `NaN`, and Wilder ATR/RSI restart after it.
If the expected last-closed higher bucket is missing, every value leaf is
`NaN` while `bar_age` stays finite and reports the staleness. Because both
resampling and the join are prefix operations, truncating later base bars does
not change an earlier MTF value.

The registry (`rcsr schema`) marks all 36 `available:true` with
`family=mtf`, `timeframe` `1h|4h|1d`, warm-up in higher bars and the PIT rule.
`docs/mtf-inventory.json` is the reviewable 36-row inventory with formula and
test id; `tests/enrich.rs` pins hand-built higher-bar values, NaN/gap behaviour,
the future-shift failure and prefix replay.

### Device MTF resample and causal join (`--features cuda`)

`src/mtf_gpu.rs` and `src/cuda/mtf.cu` move the same contract to the device.
Base rows are bucketised with `floor(t/tf)` per symbol; a deterministic
exclusive scan of the new-bucket flags gives one thread per bucket, which folds
its contiguous input rows in order (`open=first`, `high=max`, `low=min`,
`close=last`). No empty bucket is emitted: a missing hour simply leaves a hole
in the higher series, exactly like the CPU.

The eleven value leaves are evaluated on the smaller higher series with the
CPU's sequential arithmetic and the same prefix-subtraction order, including
the Wilder ATR/RSI seeds and gap resets. One monotone cursor per symbol then
produces the last-closed join index once per timeframe; each requested leaf is
a one-thread-per-base-row gather over that index. The CPU guard
`higher.time == bucket_start(base.time) - tf` is preserved, so a stale bucket
is `NaN` for every value leaf while `bar_age` stays finite and exact. A partial
or current higher bucket is never visible.

Bucket buffers are sized on the host from each symbol's hour span (capped at its
row count), which bounds the `1h`, `4h` and `1d` bucket counts for any holes. The
device accepts only timeframes that are a multiple of 3600 s. If the device path
still fails, `enrich` prints one stderr warning naming the stage and return code
and computes the 36 columns on the CPU; the output bytes are unchanged.

`tests/mtf_gpu.rs` pins the complete 36-column CPU/GPU parity (bitwise for every
leaf except `vol`, whose `sqrt` is within `1e-12`), a missing bucket for
`1h`/`4h`/`1d`, a partial trailing bucket, prefix replay at ~20 cuts and symbol
batch/chunk-size invariance. `rcsr schema` reports the family under
`gpu_feature_parity`; `rcsr enrich --help` documents the device path.

**Dispatch guard.** On the paired 1.05M-row, two-symbol 15m fixture the MTF
phase was CPU 5.08 s vs device 6.45 s (kernel 3.55 s, transfer 357 MB), so the
device path is **opt-in** with `--gpu on` until a paired 16M-row run establishes
the crossover. `--gpu auto` keeps the CPU reference for MTF and reports the
reason in the plan; the cost model for the other families is unchanged. This is
a measured guard, not a speed-up claim: the 16M paired measurement is still
missing, and the 1.05M fixture has only two symbol segments, so the
one-thread-per-segment leaf/join passes underutilise the device.

### Device MTF as-of join and `bar_age` (`--features cuda`)

`src/cuda/pit.cu` (issue #302) also joins higher-timeframe bars onto base rows
with the monotone merge used for membership; `src/panel_pit.rs` is the CPU
reference and mirrors `ohlcv::join_causal`. A higher bar `h` is visible to a row
at `t` when `h.time + tf <= t`, so a bar closing exactly at `t` is visible and
the forming bar is not. Per row:

* `bar_age = t - (h.time + tf)` in seconds; `NaN` only when no bar is visible. A
  stale bucket keeps a finite `bar_age`.
* `fresh` is 1 when the visible bar is exactly the previous expected bucket
  (`floor(t/tf)*tf - tf`). A row that is not fresh yields `NaN` value leaves, so
  a missing bucket is never bridged by an older one.
* `lookahead_violations` counts any row whose leaf was not yet closed. A column
  shifted one bar into the future is reported, with the same count as
  `ohlcv::lookahead_violations`.

The result is bit-identical to the CPU reference at any block and chunk size, and
any prefix of the timeline equals the full run truncated (`tests/panel_gpu.rs`,
`tests/panel_pit.rs`).

### Device panel preparation: duplicates, 15m → 1h and gaps (`--features cuda`)

`src/panel_prep.rs` is the exact CPU reference and `src/panel_prep_gpu.rs` /
`src/cuda/prep.cu` the fused device pass (issue #301). The input is an
**already decoded** symbol-major block; ZIP inflate, CSV parsing, unit decisions
and validation stay on the CPU exactly as before. The device pass owns four
operations in one block:

1. **Stable sort by `(symbol, time)`** inside each contiguous symbol segment.
   Equal timestamps therefore keep their pre-sort input order, which is the tie
   order the CPU reference uses for duplicate aggregation.
2. **Duplicate scan.** An equal-time run whose nine payload fields and
   `complete`/`traded` flags are bit-identical collapses to one bar with
   `dup_rows = run_len`; its flows are **not** summed. A differing run is folded
   exactly once (`open=first`, `high=max`, `low=min`, `close=last`, flows
   added left-to-right, `complete` AND, `traded` OR) and reported as
   `DUPLICATE_CONFLICT`. The JUV 2022-03 four-copy month is the reference case:
   a naive sum would quadruple its volume.
3. **Complete-bucket fold.** A bucket is complete only when it holds the four
   contiguous expected `in_step` instants (`cov15m == 4`). Otherwise it is
   dropped, or emitted under `keep_partial` with `cov15m` 1–3 and
   `PARTIAL_BUCKET`. Zero-volume buckets are kept with `traded=false`; `H == L`
   is a valid complete bucket. Empty buckets are not manufactured.
4. **Gap tags.** `gap = max(0, (t[i] - t[i-1]) / out_step - 1)` inside one
   chain. The chain resets at a listing or delisting (segment change), a
   migration and a `member=0` exclusion (caller-supplied chain id); no gap is
   bridged by a feature.

Timestamps stay `i64` UTC seconds throughout. Conflict offsets and bucket
compaction are exclusive prefix sums, and no atomic touches an output, so the
same block produces identical bytes at any launch block size. Prefix replay is
also exact: rows at or before a cut depend only on earlier input.

A bar's `dup_rows` is its equal-time run length; a bucket's `dup_rows` is the
sum over its bars, so a clean hour is 4 and an hour of the four-copy JUV month is
16 while its volume stays one copy's worth.

Measured on a 4.78M-row decoded block (200 symbols, holes, partial hours,
repeated and conflicting copies, `tests/panel_gpu.rs`): CPU reference 0.56-0.62 s;
device pass 0.022-0.030 s; upload 0.11 s; download 0.03-0.08 s (inflate and
parse excluded). The transfer dominates, so the device pass only pays when the
block is already resident or fused with the following enrichment. Panel hashes
are equal CPU vs device, cold vs warm and at block sizes 64/256/1024, for both
`keep_partial` settings.

`tests/panel_gpu.rs` pins CPU/device parity on mixed fixtures (identical and
conflicting duplicates, missing and partial buckets, zero volume, `H == L`),
the four chain resets, the JUV/AUD-like repeated-month case, conflict ordering,
NaN gaps, bit-different zero copies, prefix replay at twenty cuts and
block-size invariance. The production `prep` path still sums identical
duplicates (#259 is open); wiring this policy into `rcsr prep` is not part of
this seam. The pass does not touch parse or inflate.

## Corporate actions

`adjust(actions, bars)` returns an `AdjustedSeries` with three parallel views:
the **raw** bars, the per-bar price and volume factors, and the adjusted bars.
Splits divide historical prices by the ratio and multiply historical base
volume by it; cash dividends use `(prev_close - cash) / prev_close` on bars
before the ex-date. Both raw and adjusted series are always available.

## Symbol migrations

`Migration { old, new, k, tolerance }` and `splice(old, new, k, tolerance)`.

The boundary ratio is `old_close / new_open`. A correct `k` makes it close to
`k`; a measured `|ln(ratio/k)|` above the tolerance (default 0.5 %) is a hard
error naming both values, so TOMO's measured 1.0169 against a declared `k=1`
fails instead of contributing a 1.69 % fake return. Prices are divided by `k`,
base volume and taker-buy base multiplied by `k`, and old bars at or after the
successor's first instant are removed. `apply_migrations` applies a table to a
per-symbol map and refuses a chain or a missing leg.

## Gaps, features and labels

`is_gap_step`, `gap_buckets`, `window_gap_mask` and `label_gap_mask` make gaps
explicit. A feature window or label window that spans a missing bucket is
`NaN`, never bridged by a moving average, ATR or RSI.

`rcsr::features`:

- `ta:core`: `ret1`, `ret5`, `ma20_ratio`, `ma60_ratio`, `bb20_z`, `atr14`,
  `rsi14`, `vol20_ratio`;
- `ta:full`: core plus `natr14`, `adx14`, `di_plus`, `di_minus`, `cmf20`,
  `mfi14`, `vwap20_dev`;
- `exotic:cheap`: `rmean5`, `rstd5`, `skew20`, `kurt20`, `range1`,
  `close_pos20`;
- `candle:geometry`: `body1`, `upper_wick1`, `lower_wick1`, `open_gap1`;
- `cross:btc`: `cross_ret1`, `cross_beta20` (requires a benchmark series);
- `calendar`: `dow_sin`, `dow_cos`, `hour_sin`, `hour_cos`.

Every column reports its warm-up. Wilder indicators (ATR, RSI, ADX/DI) restart
after a gap rather than carrying state across an outage.

`LabelSpec::parse("fwd:1h,cost=10bp,volnorm=96,stride=1", tf)` and
`forward_label(bars, tf, spec)` enter at the **next open**, exit at the close
`horizon` bars later, subtract a symmetric round-trip cost (`2 * cost`), divide
by trailing volatility when requested, and drop labels whose window spans a gap.
The builder reports warm-up, gap and stride row counts.

## Streaming over `.rcsb`

`for_each_bar_from_store(path, f)` reads a column store one row group at a time
and yields bars with the usual column aliases; `bars_from_store` is the
collecting convenience. This is the bridge from `rcsr import`/`rcsr prep` to the
same bar algebra without an intermediate CSV.

## Not in this module

The `rcsr prep` CLI (#123) owns raw-archive decoding, the spec file and the
manifest. The point-in-time panel builder (#145) owns membership, costs and the
lockbox. Both call these types.

## Incremental enrichment (`rcsr enrich`, issue #200)

`rcsr prep` builds a store from raw archives. `rcsr enrich` is the other half:
it opens an existing OHLCV store (or a CSV), computes pre-computed feature
columns and writes a new store whose existing columns are copied **byte for
byte**. Rows are never rebuilt, re-sorted or re-encoded, and the input file is
never modified when `--out` names a different path.

```sh
# Appending is the default: every existing column keeps its position, dtype,
# dictionary and values. A name that already exists is an error.
rcsr enrich --data panel_hourly.rcsb --out panel_hourly_rich.rcsb \
  --features returns,ma,vol,atr,rsi,bollinger,vz,range,candle,donchian,calendar,lags,xs,btc

# Explicit replacement of one numeric column, keeping its schema position.
rcsr enrich --data panel_hourly_rich.rcsb --out panel_hourly_rich2.rcsb \
  --features rsi14 --replace

# Targets use the #125 label semantics (enter at the next open, symmetric cost).
rcsr enrich --data bars.rcsb --out bars_labelled.rcsb --label fwd:1h,cost=10bp,volnorm=96

# CSV input is loaded once into a temporary store and then follows the same path.
rcsr enrich --data panel.csv --out panel.rcsb --features ta:core,returns
```

The report (`<out>.enrich.json`, plus `--json` on stdout) records the input and
output SHA-256, every per-column hash, each column's nominal warm-up, observed
`NaN` count and first finite row, the inferred bar spacing, phase timings and
peak RSS. The append writer hashes the output as it streams the blocks to disk,
so a multi-gigabyte store is not read a second time just to checksum it.

### Catalogue

| group | columns | definition |
|---|---|---|
| `returns` | `r1` .. `r120` | `ln(close[i]/close[i-k])` when the two bars are exactly `k` steps apart |
| `ma` | `ma5 ma10 ma20 ma50 ma100 ma200` | `close / trailing mean - 1` |
| `vol` | `vol5 vol20 vol60 vol120` | population std of the 1-bar log returns |
| `atr` | `atr14 natr14` | #125 Wilder ATR in price units and percent |
| `rsi` | `rsi14` | `RSI14/50 - 1` (compact definition; the old 0..100 value is gone) |
| `bollinger` | `bb20_z bb20_bw` | z-score and `4*std/mean` |
| `vz` | `vz96` | z-score of `log1p(volume)` |
| `range` | `range1` | `(high - low) / close` |
| `candle` | `clv1 body1 upper_wick1 lower_wick1 open_gap1` | `clv1 = (C-L)/(H-L)-1/2`; `body1 = (C-O)/(H-L)`; wicks as shares of range |
| `donchian` | `don20 don96` | close position in the trailing high/low channel; `don96` is centred on 0 |
| `calendar` | `dow_* hour_* month_* dom_*` (sin/cos) | UTC calendar cycles |
| `lags` | `lag_r1_1` .. `lag_r1_10` | causal copies of `r1`, refused across a gap |
| `xs` | `xs_rank_r1 xs_rank_r20 xs_rank_vol20` | per-timestamp PIT rank in `[-1,1]`, `N>=20` |
| `btc` | `btc_r1 btc_r20 btc_vol20 rel_btc_r1 rel_btc_r20 beta_btc_60` | benchmark joins by exact timestamp |
| `cross` | 45 columns | the complete #273 cross-asset, ETH, market-state, breadth, dispersion, average-correlation and rank family |
| `py15m` | 35 columns | the Python `enrich_panel_15m.py` set, for parity |
| `mtf` | 36 columns | `mtf_<1h|4h|1d>_<12 leaves>`, higher-timeframe resample + causal close join |
| `compact` | 52 columns, compact order | the exact compact-panel regression subset at the **15 m base** |
| `flow` | `vwap` plus `taker_ratio96 taker_imb16 trade_intensity96 avg_trade_size96 amihud96 zero_ret_share96 obv_z96 cvd_proxy vwap_dev96` | the complete OHLCV/flow family (#270) |
| `price_action` | `clv16 body16 rng_atr prev_range_pos inside_s outside_s engulf_s harami_s run8 nr4 trend96` | continuous candle geometry and multi-bar structure (#261); `batr` is materialised by `compact` |
| `all` | every legacy group above except `cross`, `py15m`, `mtf`, `compact`, `flow`, `price_action` | |

The five `py15m` names `mkt_r16`, `mkt_r96`, `mkt_vol96`, `rel_r16` and
`rel_r96` predate #273. `mkt_*` are now the canonical equal-weight market
columns below; on a single-symbol slice they are `NaN` because the market
requires 20 PIT members. `rel_r16`/`rel_r96` remain the legacy BTC-relative
py15m columns. This is a deliberate correction of the old six BTC joins, not a
second implementation.

`ta:core`, `ta:full`, `exotic:cheap`, `candle:geometry`, `cross:btc` and
`calendar` keep their #125 meaning and are computed by `rcsr::features`, not
copied into this module. `cross_ret1`/`cross_beta20` are accepted and use the
`--cross-symbol` series.

### Default financial feature set and registry

The production set is the **default financial feature set**, not a single
expanded panel. It is the union of:

* the raw OHLCV/flow fields (`open high low close volume quote_volume trades
  taker_buy_base taker_buy_quote`) and identity/quality fields;
* every engine column above (`all` + `py15m` + the #125 groups + `cross:btc`),
  currently **216** distinct names in the catalogue command
  `--features all,py15m,ta:full,exotic:cheap,cross:btc`;
* the compact 52-name regression subset; 38 of its names are not in that
  catalogue command, so the complete engine inventory is **254** names
  (`enrich::engine_names()`), emitted by the same command plus `compact`;
* the 36 multi-timeframe columns (`--features mtf`), which are
  `available:true` with a tested causal join;
* cross-sectional market-state/rank and reserved derivative columns, which
  remain `available:false` with an explicit dependency.

`rcsr schema` prints two machine-readable inventories. `engine_catalogue`
contains every engine name with `family`, `warmup`, `available` and its exact
`formula`, plus `shared_names` mapping every cross-group name to its one
canonical producer. `feature_registry` contains one object per default-set
column with `name`, `family`, `timeframe`, `warmup`, `available`, `derived`,
`parity_tier`, `formula`, `formula_sha256`, `source`, `pit` and, when
unavailable, `reason`. `parity_tier` is `bitwise`, `1e-12`, `unavailable` or
`reference` (CPU-only reference in this change).
Unknown names remain hard errors; a registered-but-unavailable name fails with
its missing dependency, never an empty column. Adding a compact-only engine name
to the registry does not remove it from `engine_catalogue`; runtime projection is
the only way a requested column is omitted.

Every engine formula has one name and one definition. The cross-group names are
registered explicitly, not left to group order:

| name | canonical implementation |
|---|---|
| `rsi14`, `mfi14`, `cmf20`, `adx14`, `clv1`, `don96`, `macd_hist` | compact definition |
| `vwap_dev96`, `obv_z96`, `taker_ratio96`, `taker_imb16`, `trade_intensity96` | flow (#270) quote-volume definition |
| `bb20_bw` | `4*sd20/SMA20` (standalone and `py15m` share it) |
| `natr14` | `100*ATR14/C` (the #125 and compact producers agree) |

The #125 (`ta:*`, `exotic:cheap`, `candle:geometry`, `cross:btc`) engine names
obey the same gap policy as the rest of the catalogue: a one-bar return across a
hole is `NaN`, every rolling window whose span is not contiguous is `NaN`, and
`open_gap1`/`cross_ret1`/`cross_beta20` are missing across the boundary. This is
tested by `engine_feature_groups_are_gap_masked` and
`engine_inventory_is_causal_on_a_prefix`.

The compact profile is computed by one 15 m reference pass. Every window is
trailing, inclusive of `t`, and gap-masked; Wilder/EMA state restarts after a
gap. The compact builder used positional NumPy windows and therefore bridged
gaps; that difference is deliberate and is why a gap row can be finite in the
old compact CSV and `NaN` here. On contiguous rows the definitions agree.

`wasym1` is registered as `derived`. For `H > L`, with
`R = H-L`, `B = C-O`, `U = H-max(O,C)`, `D = min(O,C)-L`:

```
body1 = B/R
clv1  = (C-L)/R - 1/2
body1 - 2*clv1 = (H-C-O+L)/R = (U-D)/R = wasym1
```

The test `wasym1_identity_holds_on_finite_range_bars` checks the identity to
`1e-12`. Degenerate `H == L` bars follow the compact builder exactly:
`clv1 = -0.5`, `body1 = 0`, `wasym1 = NaN`, `lower_wick = NaN`; the identity is
undefined there.

### OHLCV/flow family (#270)

The `flow` group materialises the derived part of the family. The ten raw
fields are input columns and are copied through unchanged in their original
units:

`open high low close volume quote_volume trades taker_buy_base
taker_buy_quote vwap`

`vwap = quote_volume / volume` in price units. It is normally stored by
`rcsr prep` (#259); `rcsr enrich --features vwap` can also materialise it (use
`--replace` when the store already carries the column).

All windows are 15 m bars, trailing and inclusive of `t`:

| column | definition | warm-up |
|---|---|---:|
| `taker_ratio96` | `mean(taker_buy_base,96)/mean(volume,96)` | 95 |
| `taker_imb16` | `(2·mean(tb,16)−mean(volume,16))/mean(volume,16)` | 15 |
| `trade_intensity96` | `ln(1+mean(trades,96)/mean(volume,96))` | 95 |
| `avg_trade_size96` | `ln(1+mean(quote_volume/max(trades,1),96))` | 95 |
| `amihud96` | `1e6·mean(|ln(C_t/C_{t−1})|/quote_volume,96)` | 96 |
| `zero_ret_share96` | `mean(1[ln(C_t/C_{t−1})==0],96)` | 96 |
| `obv_z96` | z-score of `OBV`, `OBV_t=OBV_{t−1}+sign(C_t−C_{t−1})·volume_t` | 95 |
| `cvd_proxy` | cumulative `2·taker_buy_base−volume`, divided by its 96-bar population std | 95 |
| `vwap_dev96` | `(close−vwap96)/vwap96`, `vwap96=mean(quote_volume,96)/mean(volume,96)` | 95 |

NaN policy:

* a window whose timestamps are not one contiguous run at the inferred bar
  spacing is `NaN`; nothing is forward-filled and a listing, delisting,
  migration or outage boundary is not bridged;
* `OBV` and `CVD` cumulative state restarts after a gap;
* zero volume makes `vwap`, `vwap_dev96`, the taker ratios, trade intensity,
  Amihud and `cvd_proxy` `NaN`, never `0`;
* zero trades makes trade intensity `NaN`; `avg_trade_size96` uses the explicit
  `max(trades,1)` in the definition and therefore stays finite;
* missing raw fields fail closed with an error naming both the feature and the
  missing column.

`vwap_dev96`, `taker_ratio96`, `taker_imb16` and `trade_intensity96` also occur
in the compact subset. There is exactly **one** implementation of each name:
the quote-volume definition above. The compact regression fixture pins these
four against that definition; the legacy Python builder used a typical-price
VWAP proxy, which differs on rows where exchange `quote_volume` is not exactly
`typical_price × volume`. The four values are compared at
`abs/rel <= 1e-12` by `flow_shared_names_match_the_compact_regression_fixture`.

### Cross-asset, breadth and rank family (#273)

`--features cross` emits the complete inventory. Every name is registered
`family=cross`, `timeframe=15m`, `available=true`, with its warm-up, formula,
source and PIT rule in `rcsr schema`.

Benchmark joins (`btc_*`, `eth_*`, `rel_btc_*`, `rel_eth_*`, `beta_*`):

* use the exact same timestamp as the compact builder (`Cross::at`, no nearest
  or as-of search); a missing benchmark bucket is `NaN`, never `0` and never
  forward-filled;
* `btc_r1 r16 r20 r96 r672`, `btc_vol20 vol96`, `rel_btc_r1 r16 r20 r96`,
  `beta_btc_60 beta_btc_384`, and the same thirteen `eth_*`/`rel_eth_*`
  columns; `--cross-symbol` (default `BTCUSDT`) and `--eth-symbol` (default
  `ETHUSDT`) select the series, and `--cross-data`/`--eth-data` load an external
  store;
* beta is trailing, gap-contiguous, uses one-bar returns and is clipped to
  `[-5,5]`; the benchmark's own beta is `1` once its window is complete;
* a one-bucket timestamp shift changes the value; the test suite asserts that
  the shifted arm differs from the exact join.

Whole-panel market state, breadth and dispersion:

| column | definition | note |
|---|---|---|
| `mkt_r16`, `mkt_r96` | equal-weight mean of PIT member returns | includes self |
| `mkt_vol96` | equal-weight mean of member `vol96` | includes self |
| `breadth` | share of PIT members with `r16 > 0` | `[0,1]` |
| `dispersion` | population std of member `r16` | not annualised |
| `avg_corr` | mean over members of `corr(r_i, mkt_r1, 96)` | documented proxy for average pairwise correlation; `[-1,1]` |
| `rel_mkt16`, `rel_mkt96` | `r_i - mean_{j != i} r_j` | strictly ex-self |
| `mkt_ew_ret1`, `mkt_ew_ret16`, `mkt_breadth`, `mkt_dispersion` | legacy #269 aliases | one implementation, no duplicate formula |
| `btc_volume_share` | `BTC volume / total PIT member volume` | needs the raw `volume` column |

Every cross-sectional column requires at least **20** valid point-in-time
members at the timestamp. With `N < 20` the whole timestamp is `NaN`; a
member with a non-finite source is excluded from the count but does not make a
different member's market state `NaN`.

Ranks (`xs_rank_r1 r16 r20 vol20 amihud96`, plus the legacy `cs_rank_r60`) are
`2*(rank-1)/(n-1)-1` in `[-1,1]`. They use finite PIT members only, require
`N>=20`, and break ties by symbol name then row order. `-0.0` and `+0.0` are one
value for that tie order. `xs_rank_amihud96` inherits the #270 Amihud input
requirement (`quote_volume`); a missing raw field fails closed naming both the
rank and the field.

Point-in-time membership is a row filter, not a survivor universe. When the
store carries `member_asof`, only rows with `member_asof <= t` enter a
cross-sectional statistic; a pre-listing or post-delisting row is excluded and
its own cross-sectional output is `NaN`. Stores without `member_asof` use the
rows present at `t`, which is the same PIT rule when the store is itself a PIT
panel. Nothing is forward-filled, so a symbol's first and last bar are hard
boundaries. `docs/cross-asset-inventory.json` is the machine-readable
name/family/warm-up/PIT/test inventory.

### Rolling window statistics

Every rolling mean and standard deviation of the engine comes from block-local compensated prefix sums (`src/window_stats.rs` on the CPU, `src/cuda/win_stats.h` on the device). The two sides run the same loops in the same operation order with IEEE `+ - * / sqrt` only and no fused multiply-add, so a column without a transcendental in its chain is bit-identical.

* Each series (one symbol) keeps a Sum2 stream for `x` and a Dot2 stream, with an exact Dekker product, for `x*x` and `x*y`. The accumulation restarts at the series start, at every gap row and every 1,024 series-relative rows. Restarts only bound the rows that enter a window's rounding error; which windows are valid is still decided by `contig` and the finite counts.
* A window sum is the exact `two_sum` difference of two prefix pairs plus the compensation difference, as a double-word. A window that contains a restart is summed piece by piece, from the window end backwards.
* A mean is the double-word sum divided by the count, rounded once with one correction step. A population standard deviation is `sqrt((w*S2 - S1^2) / w^2)` with the numerator in double-word, and exactly `0` when every finite cell of the window is equal. A z-score over such a window is therefore `NaN`, not noise.
* Restarts make the 1,024-row blocks independent, so the device builds the prefixes of `ma`, `vol`, `bb_width` and `vz` with one thread per (series, block) and evaluates one row per thread. The finite-cell count is the only quantity that runs across blocks; it is an exact integer scan over the block counts. The rolling rings of `bb_z`, `ma_slope`, `acf1`, `vr4`, `jump_share` and `semivar_asym` replay the preceding block from its restart before emitting their own block, which reproduces the sequential walk exactly. `ou_hl96`, `rskew96` and `cusum_abs_z` stay sequential per series: their inputs are themselves rolling statistics, or a running float sum that cannot restart mid-series without changing bits.
* `ou_hl96` takes double-word co-moments over the three finite-count sets (both finite, `x` finite, lag finite) and `x = ln_1p((C - SMA96) / SMA96)`, which equals `ln(C) - ln(SMA96)` and keeps the device logarithm noise below 1e-12 in the output.
* The NaN policy of the window mean is an explicit parameter. `ma96` and `ma672` average the finite closes of the window (`RC_WIN_MEAN_FINITE`, the compact reference); the catalogue `maN` columns need a full window (`RC_WIN_MEAN_STRICT`). The two are deliberately different.
* `cusum_abs_z` is the absolute sum of the finite z-scores of the 96 rows of its own window, divided by `sqrt(96)`. The cumulative sum restarts at every gap row, so a window that starts at a gap row subtracts `0`, not the cumulative sum of the previous run. CPU and device apply the same rule.
* The max error against an exact reference is within 1 ulp for means and 2 ulp for standard deviations at condition numbers (`mean^2 / var`) up to about 1e10; `tests/window_stats.rs` reports the error of every rolling family, the locality of a symbol alone against the same symbol inside a 24-symbol panel, and the CPU feature-phase time.

### Compact parity fixture

`my-version/tests/fixtures/enrich/compact_15m_ohlcv.csv` is a contiguous 7,200
row (two symbols x 3,600 bars) 15 m slice with raw OHLCV/flow. The expected file
holds the compact builder's 52 columns for the last 300 hourly rows. The test
`compact_matches_the_python_builder_on_a_contiguous_slice` compares every finite
value at `abs <= 1e-12` or `rel <= 1e-12`, requires exact NaN masks, and reports
the per-feature worst error. Expected values are produced by the builder's own
Python functions on the same slice; the full 25 GB panel is not checked in.

The compact builder accumulates `rmean`/`rstd` with `np.cumsum` over the **whole symbol history**. On a 196k-row history, cancellation in `cs[i+1]-cs[i+1-w]` perturbs a 96-bar standard deviation by up to about `2e-8`, which a slice-local recomputation cannot reproduce. The engine windows are accurate to rounding, so the fixture compares at 1e-8 (`PARITY_TOL` in `tests/enrich.rs`).

### Candle/price-action family (#261)

The family has two halves. The compact candle subset is the parity/regression
fixture, computed by the single 15 m reference pass described above:

`clv1 clv4 body1 batr wasym1 lower_wick rng1 volrat gapatr don96`.

`wasym1` is the only exact identity in the set and is registered `derived`;
the other nine are materialised. The `price_action` group completes the twelve
default continuous/structure columns required by #261 (`batr` is the shared
compact name):

| column | definition | warm-up |
|---|---|---:|
| `clv16` | `mean((C-L)/(H-L) - 1/2, 16)` | 15 |
| `body16` | `mean((C-O)/(H-L), 16)` | 15 |
| `batr` | `(C-O)/ATR14` (compact definition) | 14 |
| `rng_atr` | `(H-L)/ATR14` | 14 |
| `prev_range_pos` | `(C_t-L_{t-1})/R_{t-1} - 1/2` | 1 |
| `inside_s` | `relu(1-relu(H_t-H_{t-1})/A) * relu(1-relu(L_{t-1}-L_t)/A)` | 1 |
| `outside_s` | `relu(1-relu(H_{t-1}-H_t)/A) * relu(1-relu(L_t-L_{t-1})/A)` | 1 |
| `engulf_s` | signed bullish minus bearish continuous engulfing | 1 |
| `harami_s` | signed opposite-body continuous harami | 1 |
| `run8` | `clip(consecutive same-sign close changes, -8, 8) / 8` | 1 |
| `nr4` | `R_t / min(R_{t-3..t})` | 3 |
| `trend96` | `C/SMA96 - 1` | 95 |

`R=H-L`, `B=C-O`, `A=ATR14` (Wilder, restarted after a gap). The two-bar
strengths use `pos(x)=relu(x)/(|x|+eps)`, `neg(x)=relu(-x)/(|x|+eps)` and relu
penetrations in ATR units; `engulf_s` and `harami_s` are positive for the
bullish cell and negative for the bearish mirror, and every strength is bounded
to `[0,1]`. No named single-bar flag (`doji`, `pin`, `bullish_engulf`) enters
the default set; those stay conditional event-study definitions behind the
#266 ablation/placebo gate.

**NaN policy.** The new geometry columns are strict: a bar with `H==L`, a
zero-volume bar (when the store carries volume), or a missing prior bucket makes
the feature `NaN`, never `0`. `clv16`/`body16` require all 16 cells to be finite;
`nr4` requires four contiguous strictly positive ranges; `trend96` requires a
contiguous 96-bar window. The compact names retain the documented compact
convention (`clv1=-0.5`, `body1=0`, `batr=0` on `H==L`) because there is exactly
one implementation per name; the parity test pins that choice.

The identities hold on every finite-range bar:

```
U/R = 0.5 - clv + min(0, body)
D/R = 0.5 + clv - max(0, body)
wasym = body - 2*clv
```

`rcsr enrich` reports `warmup`, `nan_rows`, `prevalence`, `first_finite_row`,
`first_finite_time` and `first_finite_iso` for every column in both the text
report (`# enrich-feature: ...`) and the JSON sidecar. The candle/price-action
registry entries record `source: "15m OHLCV bars; not a target construction
column"`, and no candle column is used to construct a label.

### Causality, gaps and warm-up

`rcsr enrich` reads one symbol at a time, so the input must contain one
contiguous run of rows per symbol (the layout written by `rcsr prep` and the
pilot hourly stores). A time-major `rcsr panel` store interleaves symbols and is
refused with an explicit message; streaming state across a time-major panel is a
follow-up. A store without a symbol column is treated as one series.

A feature at row `i` reads only bars at or before `i`. A window is valid only
when every bucket between its first and last bar is present at the inferred
(or `--bar-seconds`) spacing. A window that spans a listing, delisting,
migration or outage hole is `NaN`; nothing is forward-filled or bridged.
Wilder and EMA state restarts after a gap. Warm-up rows are `NaN`; the report
prints the nominal warm-up, the observed `NaN` count and the first finite row.
`cargo test --test enrich every_catalogue_column_is_causal_on_a_prefix` checks
that truncating the store never changes an earlier value.

`xs_rank_*` ranks each timestamp across PIT symbols, ignores non-finite source
values and non-members, breaks ties by symbol name then row order, maps the rank
to `[-1,1]`, and returns `NaN` when fewer than 20 finite members are present.
`-0.0` and `+0.0` are one tie. The pass holds one source column, the time column,
a row permutation and the rank column (about 32 B/row), never the whole panel.
`rcsr enrich` refuses the pass when that exceeds the detected hard memory limit.
On the CUDA path the `(time, symbol, row)` permutation, timestamp group ids and
segment offsets are built once on the device and reused by every rank column;
each column is a stable segmented sort by `(value, symbol, row)` with invalid
and non-member rows keyed last, so no per-column host sort runs.

Measured on 4 M rows, 100 symbols, PIT panel with non-members (NVIDIA GeForce RTX 3060, `cargo test
--release --features cuda --test enrich_gpu measure_cross_section_stage --
--ignored --nocapture`): the four `xs` rank columns take `xs_seconds` 4.02 s
with `--gpu off` and 1.75 s with `--gpu on`. The `cross` family takes 12.6 s and
8.4 s of a 235 s run. Equal-weight market state, breadth, dispersion and
`avg_corr` stay on the CPU in their original addition order, so their bits are
unchanged; the residual stage is under 4 % of the run, so a device port is
declined until a larger share is measured.

### `py15m` parity

`py15m` reproduces the 35 columns added by the Python
`arbeidsmappe/crypto-rcsr/pilot/enrich_panel_15m.py` step (EMA/MACD, extra RSI
and Bollinger windows, ATR50, stochastic, ADX/DI, OBV z-score, VWAP deviation,
Parkinson/Garman-Klass, 96-bar skew/kurt, `r192`, month/day-of-month cycles and
BTC market joins). The committed fixture
`my-version/tests/fixtures/enrich/btc_15m_py15m_expected.csv` is 400 rows of the
pilot's `panel_15m_rich.csv`, preceded by 1,000 rows of pre-roll in
`btc_15m_ohlcv.csv`; the test compares values within a pre-registered
`1e-6` absolute-or-relative tolerance. Two deliberate fixes are documented:
a gap is not bridged, and the first true range is `high - low` instead of
NumPy's `np.roll` wrap-around that reads the last close.

### Fused flow and price-action device producers (#297)

The `flow` registry family has **17** columns: the five raw flow inputs
(`volume quote_volume trades taker_buy_base taker_buy_quote`), the ten
`FLOW_NAMES`, and the two engine columns `cmf20` and `vwap20_dev`. The fused
device producer materialises the **twelve derived** columns (the ten
`FLOW_NAMES` plus `cmf20`/`vwap20_dev`); the five raw inputs are pass-through
store columns. `price_action` materialises **eleven** columns. The formulas are
exactly those in the two tables above; there is no second definition.

The device split is a **base pass plus consumers**:

| family | base kernels (once per run) | consumers |
|---|---|---|
| flow | elementwise `vwap`/per-trade/Amihud/OBV-step/CVD-net/CMF value; cumulative OBV+CVD | one per output column |
| price_action | elementwise geometry (`clv`, `body`, range, validity, true range) + Wilder ATR14 | one per output column |

The elementwise flow base reads `trades` and `taker_buy_base` **once**,
independent of the requested column count; `EnrichGpu::base_reads()` counts the
optional raw columns consumed by that pass and the test
`fused_flow_reads_optional_raw_columns_once` pins it at two for a twelve-column
request. `EnrichGpu::base_passes()` reports four base launches for the two
families instead of one pass per column (the ~6-vs-21 pass reduction of
`04-feature-enrich.md` §3.7).

Gap and missing-column policy is the CPU policy, not a near-copy:

* the device context carries a raw-column **presence bitset** and `d_run_id`;
  an absent optional column is uploaded as `NaN` and every consumer emits
  `NaN`, never `0`;
* `gap(i) = i>0 && same segment && run_id differs`, so a symbol boundary and an
  internal outage behave exactly like the per-symbol `gap_mask`; `gap[0]` is
  false, OBV/CVD restart at a gap, and Wilder state restarts at a gap;
* rolling consumers stage a tile of the source plane in shared memory and one
  thread builds the compensated running sums in CPU order (see [Rolling window
  statistics](#rolling-window-statistics)), so `rmean_py`/`rstd_py`/`rmean_strict`
  keep their exact operation order while global accesses stay coalesced;
  two-bar geometry runs one thread per row.

Parity tiers are declared per column in `feature_registry.parity_tier`:

| tier | columns |
|---|---|
| `bitwise` | `vwap`, `taker_ratio96`, `taker_imb16`, `zero_ret_share96`, `obv_z96`, `cvd_proxy`, `vwap_dev96`, `cmf20`, `vwap20_dev`, and all eleven `price_action` columns |
| `1e-12` | `trade_intensity96`, `avg_trade_size96`, `amihud96` (each takes a logarithm) |
| `reference` | every other registered column (CPU reference; no device claim in #297) |
| `unavailable` | the reserved derivative columns |

`zero_ret_share96` stays `bitwise` even though its definition mentions
`ln(C_t/C_{t-1}) == 0`: the kernel tests the exact close ratio instead of a
rounded logarithm, which is equivalent because a ratio that is not `1.0` cannot
round to a zero logarithm.

The parity fixture (2 symbols x 500,000 rows, 15 m, gaps, zero volume, zero
trades and `H == L`) reports **21 of 23 columns bit-identical**, zero NaN-mask
mismatches and max absolute difference **8.882e-16** for the two logarithmic
columns. Repeated GPU runs and `--chunk-rows` 1/65,536 hash identically, and
truncating the store leaves every earlier row bit-identical
(`fused_gpu_is_prefix_causal`), so no future bar enters a value.

**Measured cost model.** `features_gpu::{FLOW_COST, PRICE_ACTION_COST}` are
kernel-time constants measured on a 1,000,075-row, 109-symbol fixture (the
segment-parallel consumers only fill the device at a realistic symbol count):

| family | base ms/Mrow | per column ms/Mrow | total, 12/11 columns | wall incl. D2H |
|---|---:|---:|---:|---:|
| flow | 11.4 | 3.3 | 51.2 ms/Mrow | 99.6 ms/Mrow |
| price_action | 8.8 | 2.5 | 36.7 ms/Mrow | 64.6 ms/Mrow |

The wall figure is download-bound: per-column `cudaMemcpy` measured
~2.1-3.8 GB/s, which is the output-path problem of #300. Kernel time is what
`plan_groups` costs. A fused family is not dispatched by `--gpu auto` below
**32 symbol segments** (`FUSED_MIN_SEGMENTS`); `--gpu on` still forces it, which
is how the two-symbol parity fixture exercises every kernel. At 16 M rows and
109 symbols the measured rate projects to ~1.4 s for the 23 columns, in line
with the ~1.5 s candle+price-action / ~2 s flow estimate in
`04-feature-enrich.md` §6.

### Determinism and the cost model

`--chunk-rows` changes only read/compute batching; the arithmetic and output
order are fixed, so per-column hashes and the store checksum are identical for
every chunk size (tested). Cross-sectional ties are resolved by symbol and row
order.

The CPU path is always present. `plan_groups` estimates CPU time and GPU time
(kernel + transfers + the one-time resident upload) per family from rows,
output columns and operation class (elementwise, calendar trig, rolling
sum/std/extrema, Wilder scan, segmented rank, join). Cost model v2 (#300) adds
the output write explicitly:

```text
gpu_s = fixed + context(n) + h2d/BW_h2d + cells/rate_kernel
      + d2h/BW_d2h + write_bytes/BW_write
```

`BW_write` is measured, not assumed. On the shared ext4 store a 512 MiB
`dd conv=fdatasync` measured **222 MB/s** (2026-10-23); the same disk measured
about **62 MB/s** for a 1.96 GB feature-staging pass while hashing per column,
which is why the report prints raw `write_s` next to `d2h_s` and `kernel_s`.
`--write-bw <MB/s>` overrides the planner constant for another filesystem, and
`tests/enrich.rs::measure_write_bandwidth_on_a_filesystem` re-measures it
(`RCSR_BW_DIR=<dir> cargo test --release --test enrich measure_write_bandwidth
-- --ignored --nocapture`). A family is dispatched to the GPU only when a device
is usable, the store has at least one million rows, an exact kernel is
registered for those columns, and the estimated GPU time is *strictly* lower
than the CPU time. The context and source columns are uploaded once per run; a
second family does not re-upload `time/open/high/low/close/volume/
quote_volume`.

`--gpu auto` (the default) applies the cost model. `--gpu on` *forces* every
registered kernel above the row floor, which is how the parity fixture exercises
all kernels; it does not make the cost model claim a win. `--no-gpu`/`--cpu`
forces the reference path. A CPU-only build ignores the GPU arms entirely, and a
CUDA build with no device falls back with the reason in every plan row.

The CUDA kernels live in `src/cuda/enrich.cu` (resident context and the original
elementwise/scan families), `src/cuda/features.cu` (fused flow / price-action,
#297) and `src/cuda/tail.cu` (the parameterised engine tail, #298). Elementwise
families use one thread per row; rolling sums/standard deviations, Wilder/EMA
scans and labels run one thread per symbol segment in the same operation order
as the CPU, with `-fmad=false` so multiply-add association is identical.
Rolling extrema in the tail use a blocked prefix-suffix pass instead of the old
`O(n*w)` loop. Cross-sectional
ranks build one device `(time, symbol, row)` permutation per run, then use a
stable segmented radix sort by `(value, symbol, row)` for each column; PIT
non-members and non-finite values are keyed last and `-0`/`+0` share the CPU
tie order. All accumulation is f64. The report records `gpu_columns`,
`cpu_columns`, `gpu_share`, `fallback`, kernel, D2H, staging-write seconds,
transfer bytes and the charged `write_bw` in both the text log and the JSON
sidecar.

### GPU enrichment measurement (2026-10-21)

Same 3,874,994 x 11 store, RTX 3060, release CUDA build. `--dry-run` isolates the
feature phase (the append/copy is identical on both arms):

| run | features | GPU columns / share | feature phase | kernel | transfer |
|---|---:|---:|---:|---:|---:|
| `--gpu off` | returns (120) | 0 / 0 % | 40.56 s | – | – |
| `--gpu auto` | returns (120) | 120 / **100 %** | **33.89 s** | 0.224 s | 3.98 GB |
| `--gpu off` | own catalogue, no xs/btc (162) | 0 / 0 % | 52.51 s | – | – |
| `--gpu auto` | own catalogue, no xs/btc (162) | 128 / **79.0 %** | **39.59 s** | 0.245 s | 4.23 GB |

The auto plan sends `returns` and `calendar` to the GPU. Rolling sum/std,
Donchian, lags, candle and the Wilder/RSI families stay on the CPU with
`estimated GPU time is not below CPU`: their one-thread-per-segment kernels are
slower on this part, and the ten-column `lags` family does not repay the
resident-upload cost. At the time of that run `atr14`/`natr14`/`rsi14` were
#125 names with no GPU kernel and recorded that reason; the #298 tail kernels
below now register them, while the per-family context term can still keep a
single family on the CPU with that reason recorded.

A real `--features returns` run including the 4.0 GB append measured 84.3 s
(CPU) and 99.4 s (GPU) wall in one pair; the append write varied between 13.0 s
and 16.4 s and dominated the difference. The feature phase is the measured GPU
win above; end-to-end wall on a shared disk is not claimed as a speed-up.

Exactness on that 3.9M-row store (`parity_off.rcsb` vs `parity_auto.rcsb`,
120 return columns): **zero NaN-mask mismatches**, max absolute difference
**4.44e-16**, max relative difference **2.22e-16** (`r100` at row 2,703,037).
The per-column SHA-256 hashes differ for transcendentals because CUDA `log`/
`sin`/`cos` round differently from the host libm at the last bit; the
pre-registered tolerance is `1e-12` absolute-or-relative, and the parity test
(`tests/enrich_gpu.rs`) enforces it. Columns with no transcendental step are
bit-identical. The earlier 35 CPU-only columns in the full-catalogue run keep
identical hashes; the 127 GPU columns are tolerance-equal.

### 15.9M-row CPU/GPU pair (2026-10-22)

The exact 15.9M-row artifact on this machine is the pilot's
`panel_hourly_all.rcsb`: **15,893,809 rows x 56 columns**, version 1. It holds
`ts`, `symbol` and pre-computed features only (no `open/high/low/close/volume`),
so `rcsr enrich` cannot read it, and the append-only writer refuses version 1.
The missing artifact is an **OHLCV version-2 store at 15.9M rows**.

The largest enrichable cell is `panel15m.rcsb`: version 2, **16,113,108 rows x
11 columns** (109 symbols, 15-minute bars), 1.4 % more rows than the 15.9M
target. Both arms appended the complete advertised catalogue
(`--features all,py15m` = **201 columns**; output 212 columns), so the pair
covers every advertised name rather than a dry-run catalogue. The host shared
the RTX 3060 with the 10-hour `wf_hourly_gpu_10h_1` search (60-100 % SM
utilisation) and the ext4 disk with other builds, so wall time is noisy; the
per-phase seconds below are the report's own measurements.

| arm | wall | feature | xs | append | peak RSS | GPU columns / share | kernel | transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU run 1 | 900.7 s | 486.3 s | 26.2 s | 366.7 s | 922 MiB | 0 / 0 % | – | – |
| CPU run 2 | 1179.8 s | 475.3 s | 17.5 s | 669.7 s | 920 MiB | 0 / 0 % | – | – |
| GPU run 1 (`auto`) | 907.9 s | 322.3 s | 16.2 s | 551.6 s | 1195 MiB | 145 / **72.1 %** | 5.80 s | 20.56 GB |
| GPU run 2 (`auto`) | 866.3 s | 324.8 s | 20.4 s | 504.7 s | 1195 MiB | 145 / **72.1 %** | 6.99 s | 20.56 GB |

The feature phase is the measured GPU win: **323.5 s mean vs 480.8 s**, 1.49x
(49.8k vs 33.5k feature rows/s). The append is the same code over the same
bytes on both arms; it ranged from 366.7 s to 669.7 s on this shared disk and is
what decides end-to-end wall. The two-run means are **887.1 s (GPU) vs 1040.3 s
(CPU)**, 1.17x, but a single pair can tie (run 1: 907.9 s vs 900.7 s).
End-to-end throughput was 13.7k-18.6k rows/s (27.26 GB output, 40.7-74.3 MB/s
append).

The auto plan dispatched 145 of 201 columns: `returns` 120/121 (`r192` stays on
the CPU), `ma` 14/14, `calendar` 8/8 and `xs` 3/3, with zero fallbacks. The
other 56 columns stay on the CPU with their recorded reasons: no exact kernel
(`btc` 9, `py15m` 4); one-thread-per-segment kernels that do not repay the
resident upload (`atr` 8, `rsi` 3, `donchian` 2, `lags` 10); or estimated GPU
time not below the CPU after the 10.5 s context term (`bollinger` 4, `candle`
6, `vol` 9). The run uploads 20.56 GB of device traffic once; no source column
is re-uploaded per family.

Exactness on the full pair (`cpu_16m.rcsb` vs `gpu_16m.rcsb`, every one of the
201 advertised columns) is checked by the harness below: zero NaN-mask
mismatches, every finite value within the pre-registered `1e-12`, and
bit-identical columns reported per name. The two CPU runs produced the same
store SHA-256 (`a4c99c95279c...`) and the two GPU runs the same
(`5a501b777cd0...`); the GPU store differs from the CPU store only within that
tolerance (CUDA `log`/`sin`/`cos` last bits), never in schema or order.

```sh
# CPU and GPU arms (both write the full 212-column store)
rcsr enrich --data panel15m.rcsb --out cpu.rcsb --features all,py15m --gpu off --json > cpu.json
rcsr enrich --data panel15m.rcsb --out gpu.rcsb --features all,py15m --gpu auto --json > gpu.json

# Every advertised column, NaN mask and value; ignores nothing
RCSR_ENRICH_PARITY_CPU=cpu.rcsb RCSR_ENRICH_PARITY_GPU=gpu.rcsb \
RCSR_ENRICH_PARITY_SPEC=all,py15m \
  cargo test --release --features cuda --test enrich_gpu \
  advertised_catalogue_matches_between_cpu_and_gpu_on_a_large_store -- --ignored --nocapture
```

On this pair the harness compared 210 numeric columns (201 advertised plus the
9 input value columns): **83 bit-identical**, **127 within last-bit
differences**, **max absolute difference 8.882e-16**, **zero NaN-mask
mismatches**, and every value inside the pre-registered `1e-12`. The differing
columns are the `log`/`sin`/`cos` families (`returns` and `calendar`); all
input columns and every CPU-computed feature are byte-identical.

### Append throughput fixes (2026-10-22)

Two costs dominated the old append path: a `write_all` plus a SHA-256 update per
**value** while staging each feature column, and a second full read of the
output store to checksum it after the rename. The writer now stages 64 KiB
blocks and hashes as it writes, so the checksum costs one pass instead of two.
`tests/enrich.rs::append_preserves_every_existing_column_and_metadata` checks
the streamed digest against a plain re-read.

Same 3,874,994-row store, `--features all --gpu off --dry-run` (the 171-column
feature phase, same machine and load class):

| build | feature phase | total | peak RSS |
|---|---:|---:|---:|
| before | 110.4 s | 123.4 s | 335 MiB |
| after | **64.9 s** | **75.2 s** | 346 MiB |

On the 16.1M-row cell the CPU arm went from 1408.5 s to 900.7 s (23:30 -> 15:01)
with the **same output SHA-256** (`a4c99c95...`): the 27.26 GB second read
disappeared and the bytes did not change.

### Batched output arc and pinned chunked D2H (#300)

The GPU output path no longer allocates one host `Vec` and issues one download
per feature column. `EnrichGpu::batch` takes a block of feature descriptors,
launches the kernels back-to-back into one device arena, and copies the whole
block out with a single pinned `cudaMemcpy2DAsync` per row chunk (4 MiB chunks;
source and destination pitch are one full column). The host then writes the
columns sequentially to their temp files, preserving the exact per-column bytes,
hashes and output order.

The arena and its pinned mirror are capped at **64 MiB**
(`GPU_OUTPUT_ARENA_BYTES`). A block whose columns do not fit — at 16.1M rows one
`f64` column is 128 MiB, so even one column exceeds the cap — falls back to the
per-column staging path with the reason recorded. `RCSR_ENRICH_GPU_BATCH=0`
selects the per-column path for an A/B measurement. The report now carries
`gpu.kernel_s`, `gpu.d2h_s`, `gpu.write_s`, `gpu.transfer_bytes` and the
`write_bw` charged by the planner, separately.

Paired A/B on the 3,874,994-row store, `--features returns --gpu auto
--dry-run` (same input, same 118 dispatched columns, alternating order, two runs
each):

| path | feature phase (mean) | kernel | D2H | staging write |
|---|---:|---:|---:|---:|
| batched arena | **33.75 s** | 0.23 s | **0.31 s** | 31.68 s |
| per-column | 34.40 s | 0.22 s | 0.84 s | 31.60 s |

The pinned block cuts D2H by **2.7x** and the whole feature phase by about
**2 %**; it is write-bound, not transfer-bound. The 1.96 GB of temp output took
31.6 s (about 62 MB/s including per-column SHA-256), which is the floor the cost
model now exposes. No 16.1M full-catalogue paired run is claimed here: the
64 MiB arena forbids batching at that width, so that arm keeps the per-column
staging path and its 180–300 s write floor.

### Device benchmark joins, beta and cross means (#299)

`src/bench_gpu.rs` and `src/cuda/bench.cu` move the benchmark family and the
per-timestamp cross means to the device. Registration lives in
`src/enrich.rs::gpu_supports`; `rcsr schema` publishes the tiers under
`gpu_feature_parity.benchmark`; `tests/bench_gpu.rs` is the parity suite.

**Benchmark columns (30).** The 13 BTC and 13 ETH columns of #273, the legacy
`cross_ret1` / `cross_beta20`, and the compact `rel_btc96` / `beta384`. The
benchmark symbol is a small sorted `(time, value)` array (eight planes: `r1 r16
r20 r96 r672 vol20 vol96 volume`) that the CPU already derives; it is uploaded
once. Each base row binary-searches its own timestamp for **exact equality**, so a
missing benchmark bar is `NaN`, never the previous value. Tied benchmark
timestamps are resolved by the CPU map (the last row wins) before upload, so the
device sees strictly increasing times. Own k-bar returns use the CPU gate
(series-local, contiguous, `close > 0`). Beta sums its window in ascending order
with the CPU's `cov/var` and `[-5,5]` clip (`beta384` clips the covariance to
`+-0.05` first), one thread per row, no atomics. A symbol that is the benchmark
itself gets `1.0` once its window is complete, as on the CPU.

| columns | tier | why |
|---|---|---|
| `btc_*`, `eth_*` returns and vols, `cross_ret1` | `bitwise` | gather of host-computed values |
| `rel_*`, `rel_btc96` | `1e-12` | uses the asset's own `log` (CUDA `log`, not host libm) |
| `beta_*`, `cross_beta20`, `beta384` | `1e-12` | same `log`; window order and masks identical to the CPU |

**Cross means (13).** `mkt_r16 mkt_ew_ret16 breadth mkt_breadth dispersion
mkt_dispersion rel_mkt16 mkt_r96 rel_mkt96 mkt_vol96 mkt_ew_ret1 avg_corr
btc_volume_share`. One thread per timestamp group walks the group in the CPU's
`(time, symbol, row)` order (the permutation #295 builds), so the sums use the
CPU addition order and the same `member_asof <= t` mask and 20-member floor. For
an identical source column the kernels are bit-for-bit equal to the CPU
(`tests/bench_gpu.rs` asserts `to_bits` equality). The sources (`r1 r16 r96
vol96`) are themselves `1e-12` columns when the device produced them, so those
market columns are declared `1e-12`; `btc_volume_share` reads raw volume only and
is `bitwise`. `avg_corr` keeps the CPU per-segment prefix-sum form (one thread per
segment) and then reuses the group-mean kernel.

**Dispatch.** The benchmark family is planned as the `join` class with the
measured rates 0.36 M cells/s (CPU, per-row map lookup) and 100 M cells/s
(device kernel + D2H). The previous 40 M cells/s CPU figure was not measured and
made the planner reject the device; with the measured rate `--gpu auto`
dispatches the family above the 1M-row floor. The cross means follow `--gpu`:
`on`, or `auto` above the 1M-row floor. A device failure falls back to the CPU
loop for that block and is counted in `gpu_fallback`.

**Measured (RTX 3060, 16,113,108 rows, 109 symbols, `panel15m.rcsb`).**

| family | CPU | GPU | note |
|---|---:|---:|---|
| 28 benchmark columns incl. 7 beta | feature phase **1250.6 s** | **38.6 s** (kernel 1.16 s, D2H 2.62 s, H2D 0.08 s) | **32.4x**; 28/28 on device, 0 fallbacks; CPU is the per-row `BTreeMap` lookup in the beta window |
| 10 cross-mean columns | xs 22.55 s | xs 19.41 s | 1.16x; kernel 0.31 s, 3.8 GB moved; context build dominates |
| 15 lookup-only columns | feature phase **451.1 s** | **18.5 s** (kernel 0.016 s) | 24.3x; the CPU map lookup is 0.54 M cells/s, the device time is the temp-store write |

The 28-column pair has 15 `bitwise` columns with identical SHA-256 and 13
`1e-12` columns (`rel_*`, `beta_*`) with a maximum absolute difference of
**2.665e-15** (`beta_btc_60`) and identical NaN masks. All ten cross-mean hashes
are identical on this input. Columns per second on the device arm: 28 columns /
38.6 s = 0.73 columns/s end to end, of which the temp-store write is the
dominant term; the kernels alone run 28 columns in 1.16 s. The removed CPU work
is 1250.6 s of map lookups.

At 1.06M rows (24 symbols, `tests/bench_gpu.rs`) the family took CPU 27.7 s vs
device 3.5 s.

### Engine tail kernels (#298)

`src/cuda/tail.cu` owns the parameterised tail of the engine catalogue. The
kernels reuse the same resident context but take raw device pointers, so the
`RcEnrich` layout has one owner (`enrich.cu`). Registration lives in
`src/enrich.rs::gpu_supports`; `src/enrich_gpu.rs` is the Rust binding and
`rcsr schema` publishes the tier under `gpu_feature_parity.engine_tail`.

| group | columns | device producer | tier |
|---|---|---|---|
| returns / MA | `r16 r96 r192 r672 ret1 ret5 ma96 ma672 ma_slope672` | `rc_enrich_ret`, `rc_enrich_ma`, `rc_enrich_ma_slope` | bitwise (log excepted: `1e-12`) |
| extrema | `don20 don96 stoch_k stoch_d close_pos20` | blocked van Herk/Gil–Werman prefix-suffix (`O(n)`) | bitwise |
| ATR / ADX / DI | `atr14 natr14 atr50 batr gapatr volrat adx14 di_plus di_minus di_diff14` | run-serial Wilder recurrence | `1e-12` |
| Bollinger | `bb20_z bb20_bw bb50 bb50_bw bb96 bb_width96` | rolling mean/std per segment | `1e-12` |
| exotic | `rmean5 rstd5 ou_hl96 acf1 vr4 rskew96 jump_share semivar_asym cusum_abs_z` | causal compensated prefix rings | `1e-12` (`ou_hl96`, `acf1`, `vr4`, `rskew96`, `jump_share`, `semivar_asym`, `cusum_abs_z` take logarithms) |
| calendar | `tod_sin tod_cos` | integer bar-of-day phase | bitwise |

The extrema kernel computes the prefix and suffix extrema of blocks of exactly
the window width and combines one of each. It records window validity before
masking, so a non-finite bar makes the whole window `NaN`; CUDA's
`fmax(NaN, x) == x` cannot bridge an invalid bar. A window whose span is not
contiguous is `NaN` for every family, and Wilder/EMA state restarts after the
hole.

Measured on the RTX 3060, one million rows, `w=96` (cells counted as rows times
window, the old cost model's convention): the blocked kernel took **4.58 ms =
20,961 M cells/s**; the old `O(n*w)` loop took **2,976 ms = 32.3 M cells/s**.
That is a **650x** kernel improvement. The planner now charges
`RollingExtrema` at a conservative **180 M rows/s** (measured 218 M rows/s).
Registration is separate from the auto decision: a two-column `donchian` family
can still record an `estimated GPU time is not below CPU` reason at 16M rows
because cost model v2 charges the one-time context per family; `--gpu on` forces
it and `r16`/`r96`/`don96` are never rejected by the compact guard. `don96` keeps the compact centred convention (`-0.5`) while `don20` and
`close_pos20` return the uncentred close position. The CPU `donchian` now uses
the same strict high/low validity as the device kernel.

The compact guard in `gpu_supports` used to route every compact name to the
reference pass before the kernel table, which made `r16`, `r96` and `don96`
report `false` despite exact producers. Those three, plus the tail names whose
compact and standalone formulas agree, now dispatch; the remaining compact
names (for example `cmf20`, `mfi14`, `body1`) stay on the CPU because their
producer is the compact reference itself or belongs to the flow/price-action
family.

### Parity and batch-invariance harness (#303)

`tests/gpu_parity.rs` is the single gate for every device-registered enrichment column. The column set is `default_registry()` filtered by `gpu_supports`, never a hand list. Each column declares one tier in the registry (`parity_tier`, also in `rcsr schema`):

| tier | check |
|---|---|
| `bitwise` | SHA-256 of the raw f64 bytes is equal, no bit differs, NaN/Inf masks equal |
| `1e-12` | NaN/Inf masks equal and `\|a-b\| <= 1e-12 * (1 + max(\|a\|,\|b\|))` |

`1e-12` is declared for anything that evaluates `ln`, `sin` or `cos` (`r<k>`, `vol<w>`, `lag_r1_<k>`, `vz96`, `vol_z96`, calendar, the log-based exotic columns, the MTF `vol` leaf) and for the flow and benchmark columns named by #297 and #299. `ou_hl96` is `1e-12`: its window co-moments are double-word sums and its input is `ln_1p((C - SMA96) / SMA96)`, so a last-bit logarithm difference between the device and the host stays far below 1e-12 (`ou_hl96_response_to_last_bit_close_noise_is_bounded` measures the CPU response to 1-ulp close noise). Elementwise geometry, min/max, ranks, MTF lookups and recurrences evaluated in the CPU order are `bitwise`. A device column whose tier is not one of the two fails `every_device_column_declares_exactly_one_tier`.

The fixture has 24 symbols x 21,000 bars (476,730 rows) with holes, a 5 h and a 32 h outage (stale 1h/4h and 1d buckets), a NaN close, zero volume, `H == L`, tied returns, PIT non-member rows and fewer than 20 members before bar 3,000.

```sh
# Whole harness: ~9 minutes on an RTX 3060; one test at a time under the machine lock
cd my-version && RCSR_CUDA_ARCH=sm_86 cargo test --release --features cuda --test gpu_parity -- --nocapture
# Quick iteration (smaller fixture), one test by name
RCSR_PARITY_PER_SYMBOL=6000 cargo test --release --features cuda --test gpu_parity device_columns_match -- --nocapture
# Per-column numbers
RCSR_PARITY_VERBOSE=1 ...
# Known parity bugs (ignored, each asserts its column at its tier)
cargo test --release --features cuda --test gpu_parity bug_ -- --ignored
```

What the tests check: every column at its tier (`device_columns_match_the_cpu_at_their_declared_tier`, with a per-family table); cold vs warm store SHA-256; `--chunk-rows` 1 and 4,096 against 65,536 and an 8,191-row store layout (symbol batching); prefix replay at 20 cuts with a future-shift canary; panel prep (#301) and PIT as-of joins (#302) CPU vs device, cold/warm, block sizes, prefix replay and canary; the MTF family on a size its context accepts. `enrich::set_gpu_floor_for_tests(0)` lowers the 1,000,000-row `--gpu on` floor for this binary only.

Large store (not in git): `/mnt/lager/tmp/opencode/prep-measure/panel15m.rcsb` (16,113,108 rows). Each arm is its own invocation so no lock hold covers both.

```sh
export RCSR_PARITY_STORE=/mnt/lager/tmp/opencode/prep-measure/panel15m.rcsb RCSR_PARITY_WORK=/mnt/lager/tmp/eureqa-gates/parity-large
for arm in cpu gpu compare; do
  RCSR_PARITY_ARM=$arm cargo test --release --features cuda --test gpu_parity large_store -- --ignored --nocapture
done
```

The `gpu` arm also repeats the GPU run with `--dry-run` and requires identical per-column hashes. On 2026-10-30 the `cpu` arm finished (298 columns, 2579 s wall, 1603 s feature phase, 1.2 GB peak RSS); the `gpu` arm hung in the projected ring (#352; the reader thread waits for a free slot that is never returned) and was killed, so the 16.1M CPU/GPU comparison is not yet recorded.

### Measured enrichment (2026-10-21)

Machine: AMD/Intel 8-thread host, 31 GiB RAM, RTX 3060 (12 GiB), ext4 on
`/mnt/lager`, release build. Source store: `rcsr prep --resample 1h` over the
82 longest-lived pilot symbols, **3,874,994 rows x 11 columns** (310 MiB); the
3.9M-row target is met to within 0.6 %. The Python baseline is the real
`arbeidsmappe/crypto-rcsr/pilot/enrich_panel_15m.py` on the full
`panel_15m.csv` (2,006,580 rows, 67 columns).

| run | rows | columns added | wall | peak RSS | output |
|---|---:|---:|---:|---:|---:|
| Python `enrich_panel_15m.py` | 2,006,580 | 35 | 342 s | 875 MiB | 1.9 GiB CSV |
| `rcsr enrich --features py15m` | 3,874,994 | 35 | **22.1 s** | **115 MiB** | 1.4 GiB store |
| `rcsr enrich` full catalogue (no `xs`) | 3,874,994 | 168 | 105.4 s | 256 MiB | 5.2 GiB store |
| `rcsr enrich --features xs` | 3,874,994 | 3 | 8.3 s | 164 MiB | 399 MiB store |

The `py15m` comparison is per row: Rust processes 176k rows/s against Python's
5.9k rows/s (**~30x**), with 7.6x lower peak RSS, on a store 1.9x larger. The
full-catalogue run is 56.5 s of feature arithmetic, 15.8 s of append/copy,
0.2 s of reads and the rest hashing the 5.2 GiB output. `--features xs` spends
2.5 s in the cross-sectional pass; its working set is one source column, the
time column, a row permutation and the rank column (about 32 B/row).

The host was shared during later repeats; wall time is sensitive to that, user
CPU time is not. A repeat under load average 10.6 (another `rcsr` run at 5.3
cores) measured `py15m` at 37.5 s wall / 30.8 s user and the full catalogue at
153.7 s wall / 85.9 s user, with the same 35- and 168-column schemas. The
feature arithmetic stayed ~57 s; the difference was disk write contention.

Reproduce (the stores are local, git-ignored data):

```sh
# 82 symbols, resampled to 1h: 3,874,994 rows x 11 columns
rcsr prep --input raw15_82 --resample 1h --out hourly_3p875m.rcsb --jobs 8

# the new append-only enrichment
/usr/bin/time -v rcsr enrich --data hourly_3p875m.rcsb --out py15m.rcsb \
  --features py15m --json > py15m.json
/usr/bin/time -v rcsr enrich --data hourly_3p875m.rcsb --out rich.rcsb \
  --features returns,ma,vol,atr,rsi,bollinger,vz,range,candle,donchian,calendar,lags,btc --json > rich.json
/usr/bin/time -v rcsr enrich --data hourly_3p875m.rcsb --out xs.rcsb --features xs --json > xs.json
/usr/bin/time -v rcsr enrich --data hourly_3p875m.rcsb --out cpu.rcsb \
  --features returns --gpu off --dry-run --json > returns_cpu.json
/usr/bin/time -v rcsr enrich --data hourly_3p875m.rcsb --out gpu.rcsb \
  --features returns --gpu auto --dry-run --json > returns_gpu.json

# Python baseline (2,006,580 rows x 35 added columns)
PANEL_OUT=panel_15m_rich_py.csv /usr/bin/time -v python3 enrich_panel_15m.py
```
