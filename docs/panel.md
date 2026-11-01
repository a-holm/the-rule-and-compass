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

### Device duplicate scan, resample and gap tags (`--features cuda`)

The base panel is a byte contract, so the device form of the 15m → 1h step in
`src/panel_prep_gpu.rs` / `src/cuda/prep.cu` implements the same policy as the
CPU reference `src/panel_prep.rs` (issue #301). It consumes an already decoded
symbol-major block and never parses text or decides units.

* Duplicates are found after a **stable** sort by `(symbol, time)`. Identical
  copies collapse with `dup_rows = run_len`; differing copies are aggregated
  once at that instant and listed as `DUPLICATE_CONFLICT`. The conflict list is
  ordered by `(symbol, time)` and compacted by an exclusive prefix sum, never an
  atomic.
* An hourly bucket is complete only at `cov15m == 4` contiguous 15m instants.
  Partial buckets are dropped, or kept with `cov15m` 1–3 and a
  `PARTIAL_BUCKET` flag. Zero-volume buckets remain with `traded=false`.
* `gap` is `max(0, (t[i] - t[i-1]) / 3600 - 1)` within a chain. Listing,
  delisting, migration and `member=0` reset the chain; no gap is bridged.
* `i64` UTC seconds, exact OHLC/flow order and block-size-independent output are
  enforced by `tests/panel_gpu.rs`.

A 4.78M-row decoded block hashes identically on CPU and device, cold and warm.
The CPU parse/inflate path is unchanged by this seam.

`rcsr panel base --gpu on|auto` runs this pass on the device. The decoded
symbols of a wave (eight per `--jobs`) are concatenated into one block, one
segment per symbol, and folded in one launch; the base rows, `cov15m`, `gap`,
flags, `dup_rows` and the conflict ledger are built from the buckets exactly as
on the CPU, so the store is byte-identical. `auto` takes the device from one
million decoded rows per wave, `on` always, `off` and `--no-gpu` keep the CPU
path and the manifest without a `backend` block. A wave that runs on the CPU
under `auto` or `on` is named in `<out>.manifest.json`: the `backend` block
lists the requested mode, the device, symbols and rows per route, launches,
bytes moved and every fallback reason (`too_few_rows`, `no_device`, `launch`).
Symbols served from the `--resume` cache run on neither route. The cache is
valid for both. `tests/panel_v2.rs` pins CPU, device, warm and cross-cache store
hashes and the ledger lines for `--require-full-bucket` and `--keep-partial`.

### Device PIT as-of joins (`--features cuda`)

`src/panel_pit.rs` is the exact CPU reference and `src/panel_pit_gpu.rs` /
`src/cuda/pit.cu` the device merge. The membership **decisions**
(ranking, ADV tie-breaks, `member`) stay on the CPU in `build_membership`; the
device only maps every base row to the decision in force. It consumes a decoded
symbol-major block: base times, a caller chain id (migration) per row and a
sorted decision list per symbol.

* A decision is visible to a row when `as_of <= t`. A row at `as_of` sees it; a
  row one second before does not. This is `partition_point(|r| r.as_of <= t)` of
  `panel::build_panel_with_step`, including last-wins on equal `as_of`.
* One thread owns a chunk of consecutive rows: one binary search for the first
  row, then a monotone cursor. The result does not depend on the chunk size or
  the launch block size.
* Per row: the decision index (`-1` if none), `member_asof`, `keep`
  (`member = 1`), `gap = max(0, (t[i] - t[i-1]) / step - 1)` only between two
  adjacent kept rows of one segment and one input chain, and a `chain_id`
  counted from 1 inside each segment. `member = 0`, a missing decision, a
  listing/delisting (segment change) and a migration (input chain change) start
  a new chain, and nothing bridges them. `chain_id` is the value to put in
  `PrepBar.chain` of the #301 prep pass.
* Kept rows are compacted with an exclusive prefix sum. No atomic touches an
  output.
* `lookahead_violations` re-checks `as_of <= t` (and the leaf range) for any
  leaf index column. The join's own count is 0; a column shifted one decision
  into the future is reported.

`tests/panel_pit.rs` (CPU) pins the reference against
`panel::build_panel_with_step` on boundary fixtures and a random block;
`tests/panel_gpu.rs` pins device/CPU byte identity, cold/warm, block and chunk
size invariance, and prefix replay at 20 cuts (all rows with `ts <= T` equal
the full run truncated).

`rcsr panel build --gpu auto|on|off` (alias `--no-gpu`/`--cpu` for `off`) joins every
base bar against the membership decisions with this merge: one `PitBlock` with a
segment per symbol, one launch, then the panel rows are built from `kept_rows`,
`gap` and the decision `as_of`. The CPU route runs `panel_pit::join` on the same
block, so the store is byte-identical on either route. `auto` takes the device from
one million base rows, `on` always, `off` keeps the CPU join and the manifest
without a `backend` block. A CPU join under `auto` or `on` is named in the `v2`
block of `<out>.manifest.json`: the `backend` block lists the requested mode, the
device, symbols and rows per route, launches, bytes moved and the fallback reason
(`too_few_rows`, `no_device`, `launch`). `panel validate` always replays prefixes on
the CPU. `tests/panel_v2.rs` pins the store SHA for `--gpu off` against `--gpu on`
from a cold and a warm (`--resume`) base on a fixture with listings, delistings,
relisting gaps and membership changes.

Measured on a 4.78M-row block of 40 symbols (`pit_paired_4_78m_row_join_measurement`,
`--ignored`): the membership join (560 decisions) takes the CPU reference 0.19 s
(25 M rows/s, 3.94 dependent key loads per row) and the device pass 0.025 s
(188 M rows/s, 1.02 loads per row).

The device prep and PIT joins are gated by `tests/gpu_parity.rs` (CPU vs device SHA-256, cold/warm, block sizes, prefix replay at 20 cuts, future-shift canary). See "Parity and batch-invariance harness" in `docs/ohlcv.md`.

### Cross-sections use the same PIT rule

`rcsr enrich --features cross` computes equal-weight market state, breadth,
dispersion, the average-correlation proxy and cross-sectional ranks only from
rows that are live at the timestamp. The pass reads `member_asof` when the
store carries it and requires `member_asof <= t`; a pre-listing or
post-delisting row is excluded from every statistic and receives `NaN` for the
cross-sectional columns. It never injects a symbol from a later membership
decision, never forward-fills a missing timestamp, and requires at least 20
valid members. A symbol whose last bar is before the panel end therefore
cannot enter a later breadth, dispersion or rank value. See
[`ohlcv.md`](ohlcv.md#cross-asset-breadth-and-rank-family-273).

Under `--features cuda` the market columns (`mkt_*`, `breadth`, `dispersion`,
`rel_mkt*`, `avg_corr`, `btc_volume_share`) reduce each timestamp group on the
device in the same `(time, symbol, row)` order, with the same `member_asof <= t`
mask and the same 20-member floor, so a pre-listing or post-delisting row is
excluded exactly as on the CPU. The benchmark joins use exact timestamp
equality; a missing benchmark bar is `NaN`
([`ohlcv.md`](ohlcv.md#device-benchmark-joins-beta-and-cross-means-299)).

## Resumable builds

A full hourly rebuild from 25,000 monthly archives is longer than one short
step, so `build` can cache work per symbol:

```sh
rcsr panel build --data hourly_base.rcsb --membership membership.csv \
  --bar-seconds 3600 --resume work/panel-parts --out panel.rcsb
```

Each symbol writes one atomic `<symbol>.part` and a `parts.txt` index, and
`state.txt` pins the cache. The part is a fixed 96-byte binary record per row
(`ts, member_asof, gap, flags, nine f64`) behind a 44-byte header (magic,
version, bar step, the symbol's series end, row count and an FNV-1a of the
payload); the store's own SHA-256 is not needed inside each part because
`state.txt` pins it. Reading a part is `fs::read` plus a copy: no text parsing,
no float formatting, so a warm read is much cheaper than the store parse it
replaces.

`state.txt` records `version`, `step`, `store_sha256`, `membership_key` (a
SHA-256 of the loaded decisions, so re-serialising the CSV does not invalidate
anything) and `settings` (`--migrations`). `--jobs` is deliberately absent: it
changes nothing about the rows.

Three modes:

| cache | behaviour |
|---|---|
| absent | build every part from the input, then write `parts.txt` |
| warm (`state.txt` matches, `parts.txt` present) | **do not read the input at all**; read the parts and reassemble the panel |
| other inputs or an older format | with `--data` in hand the directory is rebuilt in place (every part is dropped first, so no stale row can enter); without it the warm load refuses |

A part is also validated against its recorded series end and checksum, so a
deleted, truncated or edited part is refused (`... does not match the cache
index`) or rebuilt from the input rather than silently shrinking the panel. The
resumed build is deterministic: cold and warm writes produce the same rows and a
**byte-identical store** (tests pin row equality and the SHA-256 of the written
`.rcsb`).

`rcsr panel membership --cache <dir>` does the same for the membership table,
which costs ~10 s because it reshapes the whole store. Its key is the store
SHA-256 plus every ranking setting (`--top`, `--lookback`, `--rebalance`,
`--min-history`, `--min-bars`, `--min-adv`, `--start`, `--end`). A warm run
copies the cached CSV out and prints `panel membership: cached (...)`. The store
checksum is taken from the manifest `rcsr prep` writes beside the store when
that manifest still agrees with the file's byte count; otherwise the store is
hashed (see the measured note on why re-reading 1.5 GB matters).

`rcsr prep` has the same shape of cache behind `--resume <dir>`: one atomic
spool plus a checksum sidecar per symbol, keyed by the archive SHA-256 and the
build settings. A warm prep reuses every completed symbol, so it does not
re-parse the ZIPs or recompute features; it still assembles and writes the
store. The measured stages are reported below.

## Default hourly panel v2 (#259)

The compact 52-column panel is a smoke fixture. The default research panel is a time-major hourly point-in-time panel built in stages. Each stage is a command and each stage reads only the previous stage's output.

```sh
rcsr panel snapshot --input raw_15m_all --jobs 8 --out snapshot.json
rcsr panel base --input raw_15m_all --jobs 8 --resume work/base-parts --out base_h1.rcsb
rcsr panel membership --data base_h1.rcsb --start 2021-01-01 --end 2026-07-01 \
  --top 100 --lookback 90 --min-history 90 --min-bars 60 --rebalance 30 \
  --cache work/membership-cache --out membership_hourly_pit.csv
rcsr panel build --data base_h1.rcsb --membership membership_hourly_pit.csv \
  --snapshot snapshot.json --out panel_hourly_default.rcsb
rcsr panel validate --data panel_hourly_default.rcsb --base base_h1.rcsb \
  --membership membership_hourly_pit.csv
```

`panel build` switches to the v2 builder when `--data` is a v2 base store (it carries `cov15m`). The legacy path above is unchanged for stores without it.

### P0 `snapshot`

Reads every `.zip` or `.csv` below `--input`. No network access. For each archive it records the SHA-256, byte size, mtime, row count, first and last timestamp, a timestamp unit histogram (`s`, `ms`, `us`), the number of duplicate rows, the number of conflicting duplicate instants and the number of invalid OHLC rows. The top level lists every archive with duplicate rows (`duplicates`), every refused bar (`invalid_ohlc`, with the reason and the four prices) and a unit histogram per symbol (`mixed_units`). The text is a pure function of the archives, so two runs on unchanged files are byte-identical. Embed it with `panel build --snapshot snapshot.json`.

An OHLC row is invalid when a price is not finite or not positive, `high < low`, `high < max(open, close)` or `low > min(open, close)`. `AUDUSDT 2020-11-13 04:15` (`high 0.72033 < close 0.72194`) is the known case.

### P1 `base`

Decodes the 15m archives and writes the symbol-major hourly base store `base_h1.rcsb` plus `base_h1.rcsb.manifest.json`.

* Timestamps are normalised per value (seconds, milliseconds, microseconds). `archive_unit` is 0, 1 or 2 for the unit of the bucket's first source bar.
* An invalid OHLC row is refused and listed. The hour that contained it has three bars, so it is dropped like any partial hour.
* Duplicate instants follow the `panel_prep` policy. Identical copies collapse to one bar and `dup_rows` counts the source rows. Differing copies aggregate once (open first, high max, low min, close last, flows summed), set `DUPLICATE_CONFLICT` (flag bit 2) and are listed with symbol and timestamp in the manifest.
* An hour is complete only with four contiguous 15m bars (`cov15m == 4`). A partial hour is dropped and counted (`--require-full-bucket`, the default). With `--keep-partial` it is written with `cov15m` 1 to 3 and `PARTIAL_BUCKET` (flag bit 1).
* `gap` is the number of missing hourly buckets since the previous written bucket of the symbol.
* The lockbox guard refuses an archive whose month intersects `[2026-07-01, 2027-01-01)` before it is decoded, and refuses decoded rows in it. `--interval 1m` is accepted only so that the guard fires first; the fold is defined for 15m bars. `--allow-lockbox` is the only override.
* `--resume <dir>` writes one atomic cache file per symbol keyed by the SHA-256 of its archives and the settings. A warm run decodes nothing and writes a byte-identical store.

Columns: `ts symbol open high low close volume quote_volume trades taker_buy_base taker_buy_quote cov15m gap flags traded dup_rows archive_unit`.

### Pinned symbol universe (`--symbols`)

`panel snapshot`, `panel base` and `prep` accept `--symbols <file>`: one symbol per line, `#` comments and blank lines ignored, duplicates collapsed, an empty list is a usage error (exit 2). Names compare byte-wise with the archive directory names (upper-cased ASCII). The filter runs on the file list before any archive is opened. A pinned list comes from `rcsr-acquire universe`.

The `universe` block in `snapshot.json`, `<out>.manifest.json` (base) and the `prep` manifest records:

```json
"universe": {"file": "universe_all_usdt_pit.txt", "sha256": "...", "listed": 588, "kept": 576,
             "dropped": ["AUDUSDT", "..."], "missing": ["AUSDT", "..."], "empty": ["HYPEUSDT", "..."]}
```

* `listed = kept + missing + empty`.
* `kept`: listed symbols with at least one archive.
* `dropped`: directories below `--input` (with or without archives) that are not listed. They contribute no rows.
* `missing`: listed symbols with no directory. The list is the venue universe and the archive may lag, so this is not an error.
* `empty`: listed symbols whose directory holds no archive (for example `HYPEUSDT`). They contribute no rows and are not an error.
* If no archive symbol is listed, the command fails (exit 3).

`base --resume` folds the list SHA-256 into the per-symbol cache key, so a changed list rebuilds the parts and the same list reuses them with a byte-identical store. Without `--symbols` the behaviour, the cache keys and the output bytes are unchanged and there is no `universe` block.

### P2 and P4 `membership`, `build`

`membership` is unchanged. `build` keeps the rows that are members at `ts` (same rules as above: no fill across a gap, listing, delisting or membership exclusion) and writes the time-major store sorted by `(ts, symbol)`.

Columns: `ts symbol listing_id member_asof open high low close volume quote_volume trades taker_buy_base taker_buy_quote vwap gap cov15m flags traded dup_rows archive_unit`.

* `vwap = quote_volume / volume`, `NaN` when the volume is zero.
* `listing_id` is global and numbered over the whole base series in symbol order, so it does not depend on the membership. A new listing starts after a gap of more than `--listing-gap-days` (default 30) between two base bars of a symbol, and at a declared `--relist SYMBOL@DATE`. LUNA is two listings.
* An open to previous close ratio beyond `--jump-factor` (default 5) between two adjacent bars of one listing is an undeclared level jump and fails the build with the list of jumps. Declare the break with `--relist`.
* Rows in the lockbox refuse the build unless `--allow-lockbox`.
* The manifest keeps the legacy blocks and adds `v2`: columns, the derivatives block (`{"available":false,"source":null,"columns":[]}` by default), listing counts, the jump check, the quality counters (duplicate buckets and extra rows, conflicts, invalid OHLC, partial buckets, zero-volume, venue and symbol outage buckets, max gap, `cov15m` histogram), the venue outage windows, the membership ledger check and the base manifest's quality block verbatim.

An hour is a venue outage when the missing symbols are at least `--outage-share` (default 0.5) of the symbols present or missing at that hour and at least three are missing. Other missing buckets are symbol outages. Nothing is bridged either way.

### P6 `validate`

Prints a JSON report and exits 3 when a check fails.

| id | check | needs |
|---|---|---|
| V1 | opens, required columns | |
| V2 | `(ts, symbol)` unique, ordered, `ts % 3600 == 0` | |
| V3 | every row is a real base bar | `--base` |
| V4 | OHLC, flow, taker and vwap invariants | |
| V5 | `cov15m == 4` unless `PARTIAL_BUCKET` | |
| V6 | `gap` matches the timestamp distance | |
| V7 | prefix replay: rebuilding from the base truncated at a cut reproduces every earlier row (`--replay N`, default 3) | `--base`, `--membership` |
| V8 | rank columns | not applicable |
| V9 | no undeclared level jump inside a listing | |
| V10 | store SHA-256 equals the manifest | manifest |
| V11 | zero lockbox rows | |
| V12 | each row's `member_asof` is the decision in force with `member = 1`; per-year rows equal the manifest | `--membership` |

Enriched feature parts are not part of this store. The feature registry and projection work (#260) adds them.

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

## The declared locked test

The protocol locked test (`protocol/windows.csv`, row `locked`) is declared with the data, so no evaluator depends on the caller remembering a flag.

```sh
rcsr panel build --data base.rcsb --membership m.csv --out panel.rcsb --locked-test 2025-07-01,2026-07-01
```

Evaluators refuse rows and windows from `2025-07-01` minus the label purge on, not only from `2025-07-01`: the label of a row at `2025-06-30T23:00` reads the first locked bars. A search excludes them with `--exclude 'ts >= 1751108400'` (default purge 219600 s; the refusal prints the exact value for the target spec in use). The split report carries `boundary_purge_seconds` and `boundary_purge_source` (`target_spec` or `protocol_default`).

The build writes `"locked_test": {"start", "end", "start_date", "end_date", "source"}` to `panel.rcsb.manifest.json` and `panel.rcsb.schema.json`. `rcsr panel schema` keeps it, `rcsr panel append` copies it from the target panel, `rcsr enrich` copies it from `--data` (or takes `--locked-test`) into `<out>.enrich.json`, and `rcsr ablation prepare` writes `<csv>.lockbox.json`. A CSV can carry the same object in a `<csv>.lockbox.json` sidecar. `--allow-lockbox` does not open it: it keeps its 2026H2 meaning.

One resolver (`rcsr::lockbox::resolve`) picks the range for search, `walkforward`, `ladder`, `ablation prepare` and `score`, `enrich` and `audit`:

1. `--lockbox START,END` (`--locked-test` for `enrich`). With a stored declaration it must cover it (`START <= stored start`, `END >= stored end`); otherwise exit 2. On undeclared data the flag adds to the protocol default: both ranges are guarded and the report says `"source":"flag+protocol_default"` with the default in `lockbox.also`. A flag that already covers the default reports `flag`. A derived sidecar (`enrich`, `ablation prepare`) declares the union when the two overlap, else the default.
2. The range declared with the data: `<data>.lockbox.json`, `.schema.json`, `.manifest.json` or `.enrich.json`. Two files with different ranges exit 3 and the message names both; equal ranges are fine. A `locked_test` object that does not parse exits 3.
3. The protocol default `2025-07-01..2026-07-01` for undeclared data, stdin and CSV files without a sidecar. Rows whose time column is not epoch seconds are not checked, so non-crypto data behaves as before.

`--lockbox none` is the only opt-out. It exits 2 when the data declares a range, and `walkforward` refuses it together with `--ledger` or `--locked-test`. Every report `lockbox` object carries `source` (`flag`, `store`, `protocol_default` or `none`) and `declared_in` (the files that declared a range, each with `file`, `start_date`, `end_date`). `rcsr enrich` and `rcsr ablation prepare` write the resolved range, including the protocol default, to their sidecar, so data derived from undeclared data carries the range from then on.

| command | with a declared or default range |
|---|---|
| `rcsr --data ...` (search) | exit 3 when any fit, validation or test row is in the range; the message names the side, the first row's date and the `--exclude '<time> >= START'` fix. With the rows excluded, `run.json` has `lockbox.rows_in_scope: 0`, `evaluated: false`, `source` |
| `rcsr walkforward` | exit 3 for a window touching the range, with or without `--ledger`; `--locked-test --ledger --sealed` reads it once |
| `rcsr ladder` | exit 3 for a fold touching the range; `--scheme locked --locked-test --ledger --sealed` reads it once and appends a `baseline_read` ledger line; a second read is exit 3 |
| `rcsr ablation score` | exit 3 for a fold touching the range |
| `rcsr audit` | rows inside the range are left out of every detector; `locked_test.rows_left_out` in the JSON report and the `# audit: locked test ... rows left out=N` line count them |

Rows are refused, never dropped silently, because a drop would change the split fractions without a record. With a declaration no check is skipped: `--time` is needed unless the data holds exactly one integer epoch-seconds column (the `i64` time column of a store); a time column that does not parse or is not epoch seconds, and any split row without a time stamp, is exit 3. Every run that exits 0 reports `lockbox.checked: true`. `--cv` checks every row left by `--exclude` and `--filter`, also with `--allow-lockbox`, and draws its folds from those rows only. Without a declaration, a time column that is not epoch seconds skips the row check and the report says `lockbox.checked: false`.

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
`complete` flag travel with the panel. The snapshot file is produced by
`rcsr-acquire verify` (README, "Acquiring the Binance Vision archive"); the builder
rejects anything that is not a JSON object. Its fields are `source`, `market`, `interval`,
`fetched_at`, `coverage_last`, `through`, `complete`, `files`, `verified` and an `entries` array
with the per-zip vendor URL, HTTP status, `etag`, `last_modified`, both SHA-256 values, byte
count and `verified` flag.

## Snapshot provenance (archive-only)

The public Binance Vision monthly archive is fetched with `rcsr-acquire fetch` and checked
against the vendor sibling `.CHECKSUM` with `rcsr-acquire verify`. The reference check on
2026-10-21 matched all five samples:

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
| **new PIT hourly (full archive)** | **4,775,757** | **445** | **332** | **2021-01..2026-06** | **478 MB** | **15.8 s** build (32.6 s before the binary parts) |

The new hourly panel is built from `raw_15m_all` (636 symbol directories,
24,952 monthly ZIPs, 2017-08..2026-06), not the old 109-name survivor download:

```sh
rcsr prep --input raw_15m_all --format binance --resample 1h --jobs 8 \
  --resume work/prep-parts --out hourly_base.rcsb
rcsr panel membership --data hourly_base.rcsb --start 2021-01-01 \
  --end 2026-07-01 --top 100 --lookback 90 --min-history 90 --min-bars 60 \
  --rebalance 30 --cache work/membership-cache \
  --out membership_hourly_pit.csv
rcsr panel build --data hourly_base.rcsb --membership membership_hourly_pit.csv \
  --bar-seconds 3600 --start 2021-01-01 --end 2026-07-01 \
  --resume work/panel-parts --snapshot snapshot.json \
  --out panel_hourly_pit_2021_2026h1.rcsb
```

Per-year symbol counts (union over the year, because names rotate monthly):
2021: 185, 2022: 199, 2023: 204, 2024: 196, 2025: 200, 2026H1: 169. Rows:
872,550 / 867,258 / 869,955 / 871,194 / 862,335 / 432,465. 9,544 gap buckets
inside continued series. Store SHA-256
`6caa29791e05d07b8eb1ce5af820d030a9d8e97c7eb6ca35bb3f1c2587eb2682`.
The panel stops at 2026-06-30, so its manifest lockbox block has `rows: 0`.

### Rebuild cost with both caches (2026-10-21)

Same host, 8 cores, one stage at a time through the shared build lock, empty
caches for the cold run. Prep cold parses all 24,952 ZIPs; warm reuses 586/586
symbols and 0 parsed. The panel is built from 586 binary parts and the warm run
reads the parts only (0 store parse).

| stage | cold | warm, parts resident | warm, parts partly evicted | peak RSS |
|---|---:|---:|---:|---:|
| prep 15m -> 1h, `--jobs 8` | **178.97 s** | **15.20 s** | **26.10 s** | 6.62 GB cold / 2.05 GB warm |
| membership (`--cache`) | 7.72 s | **0.01 s** | **0.03 s** | 3.23 GB cold / 8 MB warm |
| panel build, 586 parts | **15.83 s** | **10.00 s** | **14.51 s** | 3.24 GB cold / 1.46 GB warm |
| **full pipeline** | **202.6 s** | **29.8 s** | **36.1 s** | — |

Page cache and disk, from `/usr/bin/time -v`:

| run | stage | wall | FS blocks read | note |
|---|---|---:|---:|---|
| cold | prep | 178.97 s | 520,280 | 24,952 ZIPs parsed |
| cold | panel | 15.83 s | 0 | store and parts resident |
| warm 1 | prep | 26.10 s | 2,189,496 | spools and ZIPs evicted by the cold run |
| warm 1 | panel | 10.00 s | 0 | parts written by the cold run, resident |
| warm 2 | prep | 15.20 s | 0 | everything resident |
| warm 2 | panel | 14.51 s | 898,312 | ~439 MB of parts partly evicted |

A cache hit does almost no work: `panel membership: cached (store sha256
d328b6f1cb30)`, `prep: resume 586/586 symbols (100.0%), 586 reused, 0 parsed`,
`panel build: resume 586/586 symbols (100.0%), 586 part(s) reused, 0 parsed`.
The spread between the two warm runs is page cache, not different work.

All three artefacts are byte-identical across cold and warm:

| artefact | bytes | SHA-256 |
|---|---:|---|
| prep store `base.rcsb` | 1,505,748,818 | `d328b6f1cb3011cb37a71da4c4df86d7999d4826b09cd230ff5eeb2af2325b53` |
| membership CSV | 868,307 | `85501f878324372e477f32426bea83c68b8fd7c2db98555bf5de95abc8924cd6` |
| panel `panel.rcsb` | 477,591,961 | `6caa29791e05d07b8eb1ce5af820d030a9d8e97c7eb6ca35bb3f1c2587eb2682` |

The panel hash is also the pre-cache baseline, so the cache changes cost, not
bytes. The cold and warm panel manifests differ only in the input/output file
names (the recorded input and output hashes are identical).

**The <=60 s box is met with headroom on the reuse path: 29.8-36.1 s.** It is
not claimed for a cold rebuild (202.6 s, dominated by the 179 s ZIP parse), and
not for a partially populated cache (a changed store, membership or settings
rebuilds the directory in place, which costs the cold panel build again). The
binary part is what closed the gap: the old text parts cost 42.0 s to re-read
and parse, more than the 32.6 s cold parse they replaced; the 96-byte records
cost ~10-15 s for the whole warm panel including the 477 MB store write. The
membership cache removes the remaining 9.7 s, and the input store is keyed by
the SHA-256 its prep manifest already records, so a warm run does not re-read
1.5 GB to hash it (an absent or mismatching manifest falls back to hashing).

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
