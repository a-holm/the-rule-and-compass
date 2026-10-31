# 05_fix_bounded: fix it, and make the next one loud

One job: fix the cause and bound every wait so a future bug fails with a message.

## Inputs

**This run:**
- `../04_repro_red/output/repro.md`
- `../04_repro_red/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: unrelated cleanups, weaker exactness guards.

## Process

1. If an input is missing or empty, stop and say which.
2. Fix the cycle found in stage 03. Replace each unbounded wait on the path with a bounded one that names what it waited for. Never weaken an exactness guard.
3. Rerun the repro N times: write `fixed.md` with `hangs: 0/N` and the log path. Run the ring or area tests narrowly: `$G/issue$N-green.log` shows `test result: ok`.

## Outputs

- `fixed.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
f="$V/05_diagnose/stages/05_fix_bounded/output/fixed.md"; grep -qE '^hangs: 0/[0-9]+$' "$f" && grep -q 'test result: ok' "$G/issue$N-green.log" && ! grep -q 'test result: FAILED' "$G/issue$N-green.log"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
