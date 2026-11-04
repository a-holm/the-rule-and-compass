# 05_resume: unpause and report

One job: remove the pause file and report.

## Inputs

**This run:**
- `../04_priority/output/actions.md`
- `../04_priority/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: starting new work before the file is gone.

## Process

1. If an input is missing or empty, stop and say which.
2. `rm /mnt/lager/tmp/opencode/PAUSE`. Queued runs start by themselves.
3. Write `report.md` per `output-format.md`: measured numbers, actions taken, what is left to the user.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test ! -e /mnt/lager/tmp/opencode/PAUSE
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
