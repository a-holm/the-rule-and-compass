# 01_preconditions: start only when nothing is left

One job: prove that the run may start now.

## Inputs

**This run:**
- `gh issue list --label roadmap --state open --limit 200`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`
- `../../../_tools/bin-pin.sh`

Do NOT load: launching, tuning flags.

## Process

1. Open roadmap issues block the run. The user's parked issues are the only exception. List them in `precond.md` as `parked: #N #M` with the date the user allowed it.
2. The binary comes from the tip of `$BASE` (the integration branch): `$RUN` is a worktree whose `git rev-parse HEAD` equals `origin/$BASE`, built in the build slot. Pin it with `../../../_tools/bin-pin.sh <worktree> <run>/bin`. Check: `sha256sum` of `<run>/bin/rcsr` equals `<run>/bin/rcsr.sha256`, and `<run>/bin/rcsr --version` exits 0 (a pinned binary that needs its build directory failed with exit 127).
3. Nothing old runs: `ps -eo comm= | grep -cx rcsr` is 0. Never `pgrep -f`.
4. Once the checks pass, start at once. Do not wait idle.

## Outputs

- `precond.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
p="$V/07_longrun/stages/01_preconditions/output/precond.md"; for n in $(gh issue list --label roadmap --state open --limit 200 --json number -q '.[].number'); do grep '^parked: ' "$p" | grep -q "#$n\b" || exit 1; done; test "$(git -C "$RUN" rev-parse HEAD)" = "$(git -C "$ROOT" rev-parse "origin/$BASE")" && (cd "$RUN/bin" && sha256sum -c rcsr.sha256) && "$RUN/bin/rcsr" --version > /dev/null && test "$(ps -eo comm= | grep -cx rcsr)" = 0
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
