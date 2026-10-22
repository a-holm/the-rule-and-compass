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
| `xs` | `xs_rank_r1 xs_rank_r20 xs_rank_vol20` | per-timestamp cross-section, `(rank-1)/(n-1)` |
| `btc` | `btc_r1 btc_r20 btc_vol20 rel_btc_r1 rel_btc_r20 beta_btc_60` | benchmark joins by timestamp |
| `py15m` | 35 columns | the Python `enrich_panel_15m.py` set, for parity |
| `compact` | 52 columns, compact order | the exact compact-panel regression subset at the **15 m base** |
| `all` | every legacy group above except `py15m`, `compact` | |

`ta:core`, `ta:full`, `exotic:cheap`, `candle:geometry`, `cross:btc` and
`calendar` keep their #125 meaning and are computed by `rcsr::features`, not
copied into this module. `cross_ret1`/`cross_beta20` are accepted and use the
`--cross-symbol` series.

### Default financial feature set and registry

The production set is the **default financial feature set**, not a single
expanded panel. It is the union of:

* the raw OHLCV/flow fields (`open high low close volume quote_volume trades
  taker_buy_base taker_buy_quote`) and identity/quality fields;
* every engine column above (`all` + `py15m` + the #125 groups), currently 216
  distinct names;
* the compact 52-name regression subset;
* multi-timeframe context, cross-sectional market-state/rank and reserved
  derivative columns.

`rcsr schema` prints `feature_registry`, one object per column with `name`,
`family`, `timeframe`, `warmup`, `available`, `derived`, `formula`,
`formula_sha256`, `source`, `pit` and, when unavailable, `reason`. Unknown names
remain hard errors; a registered-but-unavailable name fails with its missing
dependency, never an empty column.

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

### Compact parity fixture

`my-version/tests/fixtures/enrich/compact_15m_ohlcv.csv` is a contiguous 7,200
row (two symbols x 3,600 bars) 15 m slice with raw OHLCV/flow. The expected file
holds the compact builder's 52 columns for the last 300 hourly rows. The test
`compact_matches_the_python_builder_on_a_contiguous_slice` compares every finite
value at `abs <= 1e-12` or `rel <= 1e-12`, requires exact NaN masks, and reports
the per-feature worst error. Expected values are produced by the builder's own
Python functions on the same slice; the full 25 GB panel is not checked in.

One numerical note belongs in the record: the compact builder accumulates
`rmean`/`rstd` with `np.cumsum` over the **whole symbol history**. On a 196k-row
history, cancellation in `cs[i+1]-cs[i+1-w]` perturbs a 96-bar standard
deviation by up to about `2e-8`, so a slice-local recomputation cannot reproduce
that artefact to `1e-12` even though the formula is identical. The fixture
therefore pins the formula and summation order on the same slice; the residual
against the historical full-panel CSV is recorded in issue #269.

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

`xs_rank_*` ranks each timestamp across symbols, ignores non-finite source
values, breaks ties by symbol name then row order, and returns `NaN` when fewer
than two values are present. The pass holds one source column, the time column,
a row permutation and the rank column (about 32 B/row), never the whole panel.
`rcsr enrich` refuses the pass when that exceeds the detected hard memory limit.

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

### Determinism and the cost model

`--chunk-rows` changes only read/compute batching; the arithmetic and output
order are fixed, so per-column hashes and the store checksum are identical for
every chunk size (tested). Cross-sectional ties are resolved by symbol and row
order.

The CPU path is always present. `plan_groups` estimates CPU time and GPU time
(kernel + two transfers + the one-time resident upload) per family from rows,
output columns and operation class (elementwise, calendar trig, rolling
sum/std/extrema, Wilder scan, segmented rank, join). A family is dispatched to
the GPU only when a device is usable, the store has at least one million rows,
an exact kernel is registered for those columns, and the estimated GPU time is
*strictly* lower than the CPU time. The context and source columns are uploaded
once per run; a second family does not re-upload `time/open/high/low/close/
volume/quote_volume`.

`--gpu auto` (the default) applies the cost model. `--gpu on` *forces* every
registered kernel above the row floor, which is how the parity fixture exercises
all kernels; it does not make the cost model claim a win. `--no-gpu`/`--cpu`
forces the reference path. A CPU-only build ignores the GPU arms entirely, and a
CUDA build with no device falls back with the reason in every plan row.

The CUDA kernels live in `src/cuda/enrich.cu`. Elementwise families use one
thread per row; rolling sums/standard deviations, Donchian, Wilder/EMA scans and
labels run one thread per symbol segment in the same operation order as the CPU,
with `-fmad=false` so multiply-add association is identical. Cross-sectional
ranks use a stable segmented radix sort with the CPU's `(time, symbol, row)` tie
order. All accumulation is f64. The report records `gpu_columns`,
`cpu_columns`, `gpu_share`, `fallback`, kernel seconds and transfer bytes in both
the text log and the JSON sidecar.

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
resident-upload cost. `atr14`/`natr14`/`rsi14` are still #125 names with no GPU
kernel and record that reason.

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
