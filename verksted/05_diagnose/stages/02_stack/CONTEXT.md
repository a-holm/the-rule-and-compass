# 02_stack: a backtrace of every thread

One job: capture function and line of every thread of the stuck process.

## Inputs

**This run:**
- `../01_prove_stuck/output/stuck.md`
- `../01_prove_stuck/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../scripts/stacks.sh`
- `../../CONTEXT.md`

Do NOT load: guessing the cause, code changes.

## Process

1. If an input is missing or empty, stop and say which.
2. The process must run with `RCSR_ATTACHABLE=1` (the machine runs `ptrace_scope=1`; do not change the system setting). Wrap a new run with `scripts/stall-watch.sh -i 60 -m 10 -l <log> -- <cmd>`.
3. `scripts/stacks.sh <pid> output/stacks.txt`. Exit 0 is gdb, 3 is eu-stack, 4 is the `/proc` table only. Exit 4 is not enough: relaunch attachable.

## Outputs

- `stacks.txt` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test "$(grep -cE '^#[0-9]+ ' "$V/05_diagnose/stages/02_stack/output/stacks.txt")" -ge 1
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
