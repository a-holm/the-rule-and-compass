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
