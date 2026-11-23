# 04_new_type: strategies for a dataset no standard type covers

One job: classify an unknown dataset by its structure, choose its time, split, lock, target and view rules, and build a base store with a recipe.

## Inputs

**This run:**
- `../01_intake/output/card.md`
- `../02_acquire/output/acquire.md`
- `../03_profile/output/profile.md`
- `../03_profile/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../references/type-stage-contract.md`
- `../../references/data-types.md`
- `../../references/time-rules.md`
- `references/strategies.md`
- `../../../_templates/adapt.md`
- `../../../../docs/data-format.md`
- `../../../../docs/data-preparation.md`
- `../../../../docs/time-series.md`

Do NOT load: the references of other type stages unless the strategy table sends you there, files at or after `dev_end`, sealed directories.

## Process

1. If an input is missing or empty, stop and say which.
2. Answer the six structure questions of [strategies](references/strategies.md) from `profile.md` and the card, in writing. Write every answer that is a guess into `deviations` of `adapt.md`.
3. If the answers match a `ready` row of `data-types.md` after a hashed conversion step, convert, set `type_stage` in `profile.md` to that stage, and stop here: the check below prints `fail: ... reclassified`. The run continues in that stage.
4. Otherwise choose the rules from the tables of `strategies.md`: time column and unit, entity key, spacing, lock and development end, target and horizon, purge and embargo, split scheme, feature view.
5. Build the base store: `rcsr import <file.csv> "$STORE" --time <column> --group-rows <N>` (it has no lock flag). Declare the locked test with the data in `"$STORE".lockbox.json` (shape in `strategies.md`). A table with no time column gets a held-out file; record its SHA-256 as `locked_test` in `adapt.md` (the engine reads no range for such data).
6. Run the mandatory checks of `strategies.md`: prefix causality of every derived column, `rcsr audit` on the CSV, the lock refusal drill.
7. Copy `../../../_templates/adapt.md` to `output/adapt.md`. Set `type: 04_new_type`, `variant: <short name>` and every other key. A key that does not apply takes `none` and the reason.
8. When a second dataset of this shape is expected, write `output/type-proposal.md`: one register row for `data-types.md` (type, stage name, what a source holds, what decides the variant) and the text of a placeholder issue for the new type stage.
9. Name the lessons of this run for `../09_lessons/CONTEXT.md`: every guess, every stop, every engine gap.

## Outputs

- `adapt.md` -> `output/`
- `type-proposal.md` -> `output/` (optional)
- `check.md` -> `output/`
- the base store `$STORE` and its manifest

## Check

```sh
a="$V/06_data/stages/04_new_type/output/adapt.md"; ts=$(sed -n 's/^type_stage: //p' "$V/06_data/stages/03_profile/output/profile.md"); [ "$ts" = 04_new_type ] || { echo "reclassified to $ts"; exit 1; }; for k in type variant layout bar_seconds source_bar_seconds first_bar store_first_bar approved_start dev_end locked_test series symbols raw_columns zero_volume_rows zero_volume_policy feature_groups view_excludes labels periods_per_year store deviations; do grep -qE "^$k: .+" "$a" || exit 1; done; grep -qx 'type: 04_new_type' "$a" && test -s "$(sed -n 's/^store: //p' "$a")" && { test "$(sed -n 's/^first_bar: //p' "$a")" = "$(sed -n 's/^store_first_bar: //p' "$a")" || ! grep -qx 'approved_start: none' "$a"; } && grep -qE '^files_at_or_after_dev_end: 0$' "$V/06_data/stages/03_profile/output/profile.md"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
