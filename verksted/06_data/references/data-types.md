---
type: register
description: Standard data types of the data pipeline, the type stage of each and what its source must hold.
---

# Standard data types

A type stage exists only when its row says `ready`. A `planned` row has an issue and no folder. A dataset that fits no `ready` row goes to `04_new_type`.

| Type | Stage | Status | A source of this type holds | Decides the variant |
|---|---|---|---|---|
| OHLC and OHLCV bars | `04_ohlcv` | ready | one row per bar: time, open, high, low, close; optional volume, quote volume, trade count, taker buy volume | which optional columns exist |
| trades and ticks | `04_ticks` | planned (#1362) | one row per trade: time, price, amount, optional side | whether side exists; bar size to fold to |
| order book and quotes | `04_orderbook` | planned (#1363) | snapshots or deltas: time, side, price, size, level | depth, snapshot spacing |
| derivative series as features | `04_derivatives` | planned (#1364) | funding, open interest, basis or liquidations with a publication time | publication lag per field; the book stays spot and long only |
| fundamentals and on-chain series | `04_series` | planned (#1365) | low-frequency series with an as-of or release time, revisions possible | first-release or revised values |
| generic tabular time series | `04_tabular` | planned (#1366) | a time column and numeric columns, no bar meaning | the time unit and the target column |
| cross-sectional panel without bars | `04_cross_section` | planned (#1367) | entity, period, numeric columns | entity key, period spacing |

Every type stage follows the [type-stage contract](type-stage-contract.md).
