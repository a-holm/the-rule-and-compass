# 04_ohlcv: adapt OHLC or OHLCV bars

One job: turn development bar files into a base store and write the recipe (`adapt.md`) that fits the columns the source really has.

## Inputs

**This run:**
- `../01_intake/output/card.md`
- `../02_acquire/output/acquire.md`
- `../03_profile/output/profile.md`
- `../03_profile/output/check.md` (must be `pass`)
- the specific lessons `ohlcv.md` under `lessons/` of the local area, when it exists

**Reference (every run):**
- `../../references/type-stage-contract.md`
- `../../references/time-rules.md`
- `references/formats-and-venues.md`
- `references/columns-and-views.md`
- `references/calendar-and-gaps.md`
- `references/lessons.md`
- `../../../_templates/adapt.md`
- `../../../_tools/ohlcv-view.sh`

Do NOT load: panel, enrich or split docs, other type stages, files at or after `dev_end`, sealed directories.

## Process

1. If an input is missing or empty, stop and say which.
2. Layout and venue: match the profile against [formats and venues](references/formats-and-venues.md). An unknown layout is converted by a hashed shell step to a header CSV; record the step and its sha256. Check the licence line of the card against the vendor terms.
3. Variant from `raw_columns` of the profile: no volume `ohlc`; volume only `ohlcv`; volume, quote volume, trade count and taker buy volume `ohlcv_flow`. A volume column that is zero on every row is `ohlc`.
4. Development copy: rows with `ts < dev_end` only. A source file that also holds later rows is cut by a hashed `awk` step that prints nothing. Files named at or after the lock start are never opened.
5. Calendar and gaps: apply [calendar and gaps](references/calendar-and-gaps.md). Zero-volume bars stay unless the source documents them as invented prices; record count and policy.
6. History: `first_bar` is the first row of the source. `store_first_bar` must equal it unless the project lead approved a later start in writing (`approved_start`).
7. Build the base store with the declared lock: one series `rcsr prep` (flags in [formats and venues](references/formats-and-venues.md)); a directory of archives for a panel `rcsr panel snapshot` then `rcsr panel base` with `--locked-test` and `--lock-record` (GPU steps through the lock).
8. View: `verksted/_tools/ohlcv-view.sh --variant <variant> --series <single|panel>` prints `feature_groups` and `view_excludes` from [columns and views](references/columns-and-views.md). Compare with the dry run of `rcsr enrich`.
9. Labels and annualisation: choose horizon, stride (at least the horizon in bars, one `anchor` for every symbol) and fee per [time rules](../../references/time-rules.md); long only, benchmark cash at 0, buy-and-hold or equal-weight, never an interest rate. Compute `periods_per_year` from the bar spacing, not from a default.
10. Fill `adapt.md` (every key of the contract), with `store_first_bar` read from the store manifest.

## Outputs

- `adapt.md` -> `output/`
- `check.md` -> `output/`
- the base store `$STORE` and `$STORE.manifest.json`

## Check

```sh
a="$V/06_data/stages/04_ohlcv/output/adapt.md"; for k in type variant layout bar_seconds source_bar_seconds first_bar store_first_bar approved_start dev_end locked_test series symbols raw_columns zero_volume_rows zero_volume_policy feature_groups view_excludes labels periods_per_year store deviations; do grep -qE "^$k: .+" "$a" || exit 1; done; grep -qE '^variant: (ohlc|ohlcv|ohlcv_flow)$' "$a" && { test "$(sed -n 's/^first_bar: //p' "$a")" = "$(sed -n 's/^store_first_bar: //p' "$a")" || ! grep -qx 'approved_start: none' "$a"; } && test -s "$STORE" && test -s "$STORE.manifest.json" && { ! grep -qx 'variant: ohlc' "$a" || grep -qE '^view_excludes: .*flow' "$a"; }
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
