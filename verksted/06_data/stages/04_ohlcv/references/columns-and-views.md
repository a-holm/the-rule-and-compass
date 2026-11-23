---
type: reference
description: Which feature families each raw column supports, and the feature view of each bar variant.
---

# Columns and views

The variant is decided by the columns the source holds, not by the columns the store holds: `rcsr prep` writes a `volume` column of zeros when the source has none, and marks every such bar `traded=false`. A source volume column that is zero on every row is `ohlc`.

| Variant | Source holds | `feature_groups` and `view_excludes` |
|---|---|---|
| `ohlc` | time, open, high, low, close | the volume-free names of `ta:full` spelled out, `exotic:cheap`, `candle:geometry`, `mtf`, `price_action`, `trend`; excludes `flow`, every name in [volume-names.txt](volume-names.txt) and the raw volume fields |
| `ohlcv` | plus volume | as `ohlc` plus the names that read volume only (`vol20_ratio`, `cmf20`, `mfi14`, `vz96`, `obv_z96`); excludes the names that need quote volume, trades or taker buy |
| `ohlcv_flow` | plus quote volume, trade count, taker buy base | `ta:full`, `exotic:cheap`, `candle:geometry`, `mtf`, `price_action`, `trend`, `flow`, `vz96`; excludes nothing |

`verksted/_tools/ohlcv-view.sh --variant <variant> --series <single|panel>` prints the exact `feature_groups` and `view_excludes` for `adapt.md`. The groups `ta:core` and `ta:full` hold volume readers (`vol20_ratio`, `cmf20`, `mfi14`, `vwap20_dev`), so `ohlc` and `ohlcv` name the surviving columns instead of the group. `ohlcv-view.sh --variant <variant> --series panel --names a,b,c` checks single names and exits 1 with `refused: <name>` for each that reads a missing field.

Panel (`series: panel`): `cross`, `cross:btc` and the rank columns need a benchmark series and enough members per timestamp (floors below). On a single series every market column is NaN: leave the groups out. The group `cross` holds `btc_volume_share` and `xs_rank_amihud96`, which read volume: on `ohlc` and `ohlcv` the view uses `cross:btc,xs,btc`, or with `--schema <output of rcsr schema>` the members of `cross` that the variant supports, spelled out.

Raw fields each volume column reads: `vwap`, `vwap_dev96` and `vwap20_dev` volume and quote volume; `taker_ratio96`, `taker_imb16`, `cvd_proxy` volume and taker buy base; `trade_intensity96` volume and trades; `avg_trade_size96` quote volume and trades; `amihud96` and `xs_rank_amihud96` quote volume; `obv_z96`, `cmf20`, `mfi14`, `vz96`, `vol_z96`, `vol20_ratio`, `btc_volume_share` volume. `zero_ret_share96` reads none. A name with `vol` is usually volatility (`vol96`, `vol_ratio` is `vol96 / vol672`) and is safe on `ohlc`.

[volume-names.txt](volume-names.txt) is the list the tool and the check of stage 06 apply, one bare name per line. `ohlcv-view.sh --derive <output of rcsr schema>` prints the registry names whose formula text reads volume, `V`, `V_t`, `quote_volume`, `trades`, `tb`, `taker`, `MFI` or `OBV`; the file holds those names plus `xs_rank_amihud96`, which ranks `amihud96`. A test keeps both equal. Until the engine records the raw columns of a source, this list is the only exclusion record.

Rules
- Use the full default set the variant supports. A column is left out only because its raw field is absent, never because it looks useless.
- Run `rcsr enrich ... --dry-run --json` first and compare its column list with the view; a view name the dry run lacks is added as an explicit name or dropped from the view with a line in `deviations`.
- Calendar columns are their own family: a formula that reads one is compared with its calendar-free twin.
- Derivative columns (funding, open interest, basis, liquidations) are `available: false` until a source exists; a request is refused, never zero-filled.
- Label columns are never inputs. Labels are long only: `--label long:<horizon>,fee=<bp>,stride=<n>[,anchor=<date>]`; `fee=` is required (`fee=0` for none) and `stride` must reach the horizon in bars, or the build exits 3 unless `--allow-overlap` is passed.

Cross-section floors (panel)
- A market mean, share or dispersion column is NaN at a timestamp with fewer finite members than `--min-cross-members`; a rank or average-correlation column uses `--min-cross-members-rank`. Both default to 20 and accept 5 to 200.
- Count finite members per split group on the built base before choosing: early history often has fewer than 20 series, and the default then blanks every market column of that period and removes it from every view that uses one.
- One floor for both classes keeps the same rows in every view. A rank over 5 members takes 5 values: record that it is coarse.
- A floored build is two enrich calls: stage A is the view without the cross-section columns plus the labels; stage B appends the cross-section columns with the floors. A read store uses the same two calls, because the column order follows the calls. Floors are never mixed: `enrich` and `panel append` refuse a store with other floors (exit 2).
- After a venue-wide outage every member's k-bar return is NaN for k bars. That shortfall grows with the lookback and no floor changes it.

Feature windows count bars, not hours: `r96` is 96 bars at the store spacing. State the spacing next to every view.
