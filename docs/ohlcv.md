# OHLCV market-data layer

`rcsr::ohlcv` (issue #125, first increment) gives market data a typed shape and
makes the two classic look-ahead traps explicit: partial bars and
multi-timeframe joins.

## Types

- `Bar { time, open, high, low, close, volume, quote_volume, trades,
  taker_buy_base, taker_buy_quote, complete, traded }`. `time` is the bucket
  start in seconds since the Unix epoch (UTC). `vwap()` is
  `quote_volume / volume` when base volume is positive.
- `Timeframe`: `1m 3m 5m 15m 30m 1h 2h 4h 1d 1w`, epoch-aligned UTC buckets.
  Exchange calendars/sessions are not part of this increment.

## Resampling

`resample(&bars, tf)` requires strictly increasing input and aggregates
`open=first`, `high=max`, `low=min`, `close=last`, and sums for volume, quote
volume, trades and taker-buy. Empty buckets are **not** created: gaps remain
gaps. A zero-volume bucket is kept and flagged `traded=false`; dropping it would
bias the sample. No forward fill ever happens.

`resample_as_of(&bars, tf, now)` keeps the bucket containing `now` but marks it
`complete=false`, and emits nothing after it. A live strategy must treat that
bar as tentative.

## Causal multi-timeframe joins

`join_causal(base, higher, higher_tf)` gives every base bar the last higher bar
whose bucket has **closed** at or before the decision instant:

```text
visible <=> higher.time + higher_tf.seconds() <= base.time
```

So a 15m decision at 10:00 sees the 09:00 hourly bar, not the 10:00 one, and
sees the previous day's daily bar, never the current day's close. The join
refuses a higher series that is not aligned to its timeframe (an unaligned bar
has no well-defined close instant).

`lookahead_violations(base, joined, higher_tf)` reports rows whose candidate
bar would not have closed; a correct join returns an empty vector, and a
deliberately future-shifted column is caught. The test suite builds exactly such
a shifted column.

## Not in this increment

- Calendars: NYSE/LSE sessions, holidays, half-days, DST, RTH vs extended hours.
- Corporate actions and split/dividend adjustment; symbol migrations.
- Feature/label libraries and the `rcsr prep --mtf` CLI surface.
- Streaming resampling over an out-of-core store.

The `prep` command (#123) already resamples raw files; these types are the seam
it and the MTF CLI will share.
