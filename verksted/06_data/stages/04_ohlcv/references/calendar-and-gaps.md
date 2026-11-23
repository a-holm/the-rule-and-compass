---
type: reference
description: Calendar, resampling, gap, zero-volume, delisting, relisting and migration rules for bar data.
---

# Calendar and gaps

Details: [OHLCV layer](../../../../../docs/ohlcv.md), [panels](../../../../../docs/panel.md).

- Calendar: crypto trades every UTC day; intraday buckets are epoch-aligned UTC. Session markets (NYSE, LSE) name their calendar; a bar never crosses a session boundary.
- Resampling: open first, high max, low min, close last, sums for the volume fields. An hourly bar of the default panel needs four contiguous 15-minute bars; a partial bucket is dropped or flagged, never completed.
- Gaps stay gaps. Nothing is forward-filled or interpolated. A feature or label window that spans a missing bucket is NaN; Wilder and EMA state restarts after a gap.
- Zero-volume bars: kept and flagged `traded=false`. Dropping them biases the sample. Some venues write a bar with zero volume that repeats the previous price; that is the same convention and the bars stay. Drop them only when the source documents invented prices, and record the count. An OHLC-only source has `traded=false` on every bar after `rcsr prep`: that flag carries no information there.
- Duplicates and conflicts are counted by the base build; read the counts from the manifest, not from the rows.
- Delisting: a series ends at its last bar. The universe file lists every symbol that ever traded in the development range, also the delisted ones. Today's listing is never the universe.
- Relisting and level breaks: `panel build --jump-factor` names a symbol and time whose price jumps; declare it with `--relist SYMBOL@TIME` and rerun. No value is bridged. Record the final list in the dataset manifest.
- Migrations and redenominations: `--migrations OLD->NEW:k:tol`; a boundary ratio off by more than the tolerance is an error, not a return.
- Exclusions: leveraged tokens, stablecoins, fiat pairs, tokenised stocks and commodity tokens are excluded by a pinned list, hashed with the universe.
- Universe membership is point in time: ranked on trailing data only (`panel membership --top N --lookback D --min-history D --rebalance D`), decided before the bar it applies to.
- Annualisation follows the label, not the bar: `periods_per_year = 31536000 / max(horizon_seconds, stride_seconds)`. Hourly bars with a 24-bar horizon give 365 periods, not 8760.
