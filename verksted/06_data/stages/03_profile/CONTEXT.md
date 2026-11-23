# 03_profile: names, headers and counts decide the type

One job: profile the development files without printing a value row and name the type stage that runs.

## Inputs

**This run:**
- `../01_intake/output/card.md`
- `../02_acquire/output/acquire.md`
- `../02_acquire/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../references/data-types.md`
- `../../CONTEXT.md`
- `../../../_tools/data-profile.sh`

Do NOT load: files dated at or after `dev_end`, sealed directories, type-stage references.

## Process

1. If an input is missing or empty, stop and say which.
2. Run `verksted/_tools/data-profile.sh --dir "$DATA" --dev-end <dev_end of the card> --out output/profile.md`. It reads file names, the header line, column counts, row counts, the first and last time stamp, the median step and the count of zero values per numeric column. It refuses a file whose name carries a date at or after `dev_end` and prints no value row.
3. Classify against `data-types.md`: the columns present decide the type; the optional columns decide the variant.
4. `profile.md` holds `type_stage_suggestion:` (`04_ohlcv` when the header names open, high, low and close, or a headerless table has a documented bar layout, else `04_new_type`). Confirm or override it in `type_stage: 04_<name>`, and give the reason for an override. No `ready` row fits: `type_stage: 04_new_type`.
5. Compare `first_ts` with `first_bar` of the card. A difference goes into `profile.md` as `first_bar_delta` and stops the run until the card is corrected.

## Outputs

- `profile.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
p="$V/06_data/stages/03_profile/output/profile.md"; grep -qE '^type_stage: 04_[a-z_]+$' "$p" && test -d "$V/06_data/stages/$(sed -n 's/^type_stage: //p' "$p")" && grep -qE '^files_at_or_after_dev_end: 0$' "$p" && grep -qE '^first_bar_delta: 0$' "$p"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
