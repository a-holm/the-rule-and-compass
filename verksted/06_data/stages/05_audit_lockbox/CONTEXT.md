# 05_audit_lockbox: no look-ahead, no lockbox peeking

One job: audit for leakage and show that the 2026H2 lockbox is refused.

## Inputs

**This run:**
- `../04_enrich/output/enrich.md`
- `../04_enrich/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/time-series.md`
- `../../../../docs/leakage.md`
- `../../CONTEXT.md`

Do NOT load: using the lockbox for tuning.

## Process

1. If an input is missing or empty, stop and say which.
2. `rcsr audit --data "$ENRICHED" --target "$TARGET" --time ts --horizon N --json`. Exit 5 is a failed audit. Then the copy-paste audit of `docs/time-series.md`.
3. Run a test window that overlaps 2026H2 without `--allow-lockbox`: it must be refused. `--allow-lockbox` is for the declared final test only.
4. Write `audit.md` with the lines `audit_exit: <code>`, `split_mode: <mode>`, `gap: <n>` and `lockbox_refusal_exit: <code>`.

## Outputs

- `audit.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
a="$V/06_data/stages/05_audit_lockbox/output/audit.md"; grep -qx 'audit_exit: 0' "$a" && ! grep -qx 'split_mode: random' "$a" && grep -qE '^lockbox_refusal_exit: [1-9][0-9]*$' "$a" && ! grep -qi 'gap.*ignored' "$a"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
