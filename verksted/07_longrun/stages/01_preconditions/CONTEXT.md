# 01_preconditions: start only when nothing is left

One job: prove that the run may start now.

## Inputs

**This run:**
- `gh issue list --label roadmap --state open --limit 200`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: launching, tuning flags.

## Process

1. Open roadmap issues block the run. The user's parked issues are the only exception. List them in `precond.md` as `parked: #N #M` with the date the user allowed it.
2. The binary is built from the tip of `main`: `$RUN` is a worktree of `main` and `git rev-parse HEAD` equals `main`.
3. Nothing old runs: `ps -eo comm= | grep -cx rcsr` is 0. Never `pgrep -f`.
4. Once the checks pass, start at once. Do not wait idle.

## Outputs

- `precond.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
p="$V/07_longrun/stages/01_preconditions/output/precond.md"; for n in $(gh issue list --label roadmap --state open --limit 200 --json number -q '.[].number'); do grep '^parked: ' "$p" | grep -q "#$n\b" || exit 1; done; test "$(git rev-parse HEAD)" = "$(git -C "$ROOT" rev-parse main)" && test "$(ps -eo comm= | grep -cx rcsr)" = 0
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
