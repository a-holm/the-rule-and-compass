---
type: reference
description: File layouts, time units and known quirks of bar sources, and how to detect each.
---

# Bar formats and venues

Detection order: header names, then column count, then the time unit. `rcsr prep --format auto` (the default) applies the same order.

| Layout | Detected by | Columns | `--format` |
|---|---|---|---|
| header CSV | a header row with `time` or `ts`, `open`, `high`, `low`, `close` (aliases: `timestamp`, `date`, `datetime`, `open_time`, `o`, `h`, `l`, `c`, `vol`, `v`, `qv`, `count`, `taker_buy_base`) | time and OHLC required; volume, quote volume, trades, taker buy base and quote optional | `ohlcv` |
| Binance kline archive | no header, 12 columns | open time, OHLC, volume, close time, quote volume, trades, taker buy base, taker buy quote, ignore | `binance` |
| six or seven columns, no header | no header, 6 or 7 columns | time, OHLC, volume | `bitstamp` |
| raw trades | time, price, amount | not bars | fold to bars first (below) |

Build commands (flags checked against `--help` of a current binary)
- One series: `rcsr prep --input <file|dir> --format <layout> --symbol <SYM> --resample <tf> --locked-test <START,END> --out "$STORE"`. `--calendar 24x7|nyse|lse` and `--tz UTC|+HH:MM|<zone>` name the calendar and the zone of the source. `--dry-run` prints the audit and writes nothing: run it first. `--symbols <file>` keeps the listed symbols only.
- Many symbols as the default hourly panel: `rcsr panel snapshot --input <dir> --out <snapshot.json> --lock-record <record>`, then `rcsr panel base --input <dir> --out <base.rcsb> --locked-test <START,END> --lock-record <record>`. The input directory holds `SYMBOL/SYMBOL-15m-YYYY-MM.zip`; the base folds 15-minute bars to hourly bars.
- `rcsr prep` refuses a row with `high < low` or a non-positive open or close (exit with the file and line). It counts duplicates, non-monotonic rows, zero-volume rows and converted time units in the audit and in `<out>.manifest.json`.

Time
- Stores hold UTC seconds as `i64`. `rcsr prep` converts milliseconds and microseconds and reports the unit it saw. One archive series can change unit over time (the Binance spot archive switches from milliseconds to microseconds for data from 2025-01-01); check `unit_converted` in the audit line of every build.
- `ts` is an accepted time alias; `ts_ms` is not: rename the column in a hashed step.
- Local-time sources name their zone at intake (`--tz`); the store is UTC.

Trades to bars
- `rcsr prep` has no trades layout. Sort by time, then fold with a hashed `awk` step: open first, high max, low min, close last, volume summed, bucket `floor(ts / S) * S`. Print volume with enough digits (`%.10g`); the default `awk` output format keeps six significant digits.

Cutting a development copy
- `rcsr prep` has no end-date filter and no `--lock-record`. Cut with `awk -F, -v e=<dev_end epoch> 'NR == 1 || $1 < e'` into a new file, hash it, and pass `--locked-test` to `prep`; the record enters at `enrich`.

Known sources (first bar of the history, for the maximum-history check)

| Source | First bar (UTC) | Carries | Notes |
|---|---|---|---|
| Binance spot klines (data archive) | 2017-08-17 | OHLCV, quote volume, trades, taker buy | monthly and daily zips with a vendor `.CHECKSUM`; licence below |
| Bitstamp BTC/USD hourly | 2011-08-18 | OHLCV | zero-volume bars repeat the previous price |
| Coinbase BTC-USD hourly | 2015-07-20 | OHLCV | |
| Coinbase ETH-USD hourly | 2016-05-18 | OHLCV | |
| Mt.Gox BTC/USD trades | 2010-07-17, ends 2014-02-25 | trades | fold to bars; a closed venue is its own arm, flagged |

Check the licence and the first bar again at intake: both can change. Spot only: futures and perpetual files are never staged.

Licence of the Binance archive
- Binance Vision Dataset Terms (https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md, version 1.0, last updated 2026-08-26): the datasets are provided under CC BY-NC-SA 4.0. Permitted uses are academic research and personal non-production backtesting. Live trading, redistribution of the data or its feeds and any commercial use are excluded; a redistributed derivative work credits Binance Vision and keeps CC BY-NC-SA 4.0. Raw files never enter git.
- Record the terms URL and version in the dataset card at intake. Other sources: record the licence the vendor states, or `unknown` and stop for the project lead.
