# 07_audit_lock: no look-ahead, no lock peeking

One job: audit for leakage and show that the declared locked test is refused and its rows are absent from every sidecar.

## Inputs

**This run:**
- `../06_enrich/output/enrich.md`
- `../06_enrich/output/check.md` (must be `pass`)
- the `adapt.md` of the type stage that ran

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/time-series.md`
- `../../../../docs/leakage.md`
- `../../references/time-rules.md`
- `../../CONTEXT.md`

Do NOT load: using the lock for tuning, any value from inside the locked range.

## Process

1. If an input is missing or empty, stop and say which. The recipe is `../04_*/output/adapt.md`; its `locked_test` is the range below.
2. `rcsr audit --data "$ENRICHED" --target "$TARGET" --time ts --horizon N --json`. Exit 5 is a failed audit. Then the copy-paste audit of `docs/time-series.md`.
3. `rcsr lock check --record <record> <store dir>`: exit 0 means no sidecar holds an entry inside the range; exit 4 is a finding.
4. Refusal drill: `rcsr walkforward --data "$ENRICHED" --target <return column> --expr <one column name> --time ts --window <label,train_start,train_end,test_start,test_end>` with a test window that reaches `START - purge`. It must exit 3. Never pass `--allow-lockbox`: that flag is for the declared final read only.
5. Trim drill: `rcsr protocol cpcv --start <first_bar date> --end <END date> --groups N --test-groups K --purge-seconds P --vars <view columns> --bar-seconds B --data "$ENRICHED" --out <dir>`. The last row of `groups.csv` must end before `START - P`.
6. A dataset with `locked_test: none` skips steps 3 to 5, records the reason and writes the SHA-256 of the held-out file as `held_out_sha256:`.
7. Write `audit.md` with the lines `audit_exit:`, `lock_check_exit:`, `refusal_drill_exit:`, `purge_seconds:`, `last_group_end:` (ISO UTC) and `split_mode:`. Print no value from inside the lock.

## Outputs

- `audit.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
a="$V/06_data/stages/07_audit_lock/output/audit.md"; t=$(ls "$V"/06_data/stages/04_*/output/adapt.md); k() { sed -n "s/^$1: *//p" "$2" | head -1; }; l=$(k locked_test "$t"); test "$(k audit_exit "$a")" = 0 && ! grep -qx 'split_mode: random' "$a" && case $l in none*) grep -qE '^held_out_sha256: [0-9a-f]{64}$' "$a" ;; *) test "$(k lock_check_exit "$a")" = 0 && test "$(k refusal_drill_exit "$a")" = 3 && test "$(date -u -d "$(k last_group_end "$a")" +%s)" -lt "$(( $(date -u -d "${l%%,*}" +%s) - $(k purge_seconds "$a") ))" ;; esac
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
