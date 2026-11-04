# 03_cleanup: path by path, only mine

One job: remove this ticket's worktree and branches and nothing else.

## Inputs

**This run:**
- `../02_close/output/closed.md`
- `../02_close/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`
- `../../../_tools/wt-done.sh`

Do NOT load: other agents' worktrees, bulk deletion, rm -rf.

## Process

1. If an input is missing or empty, stop and say which.
2. Confirm the ticket's agent is finished: its PR is `MERGED` and no process runs under its worktree.
3. Run `verksted/_tools/wt-done.sh "$N"`. It refuses unless the worktree is clean, no process runs in it and the PR is `MERGED`. It removes the worktree, the local branch and the remote branch of this ticket only, then prunes. No `rm -rf`, no `--force`, no other path.
4. Write `report.md`: the paths removed and the final issue state.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
! git worktree list --porcelain | grep -q "worktrees/issue-$N-" && test -z "$(git branch --list "issue/$N-*")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
