# 05_restart_check: the repository is intact

One job: prove the repository and the machine are in the recorded state.

## Inputs

**This run:**
- `../04_handoff_file/output/handoff.md`
- `../04_handoff_file/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: continuing work before all four checks pass.

## Process

1. If an input is missing or empty, stop and say which.
2. Four checks: the branch sha equals the remote sha, the worktree is clean, no cargo or rustc process runs, no `PAUSE` file exists.
3. Write `report.md`: each check with its command and result.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test "$(git -C "$RUN" rev-parse HEAD)" = "$(git -C "$RUN" rev-parse '@{u}')" && test -z "$(git -C "$RUN" status --porcelain --untracked-files=no)" && test "$(ps -eo comm= | grep -cE '^(cargo|rustc|rcsr-bench)$')" = 0 && test ! -e /mnt/lager/tmp/opencode/PAUSE
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
