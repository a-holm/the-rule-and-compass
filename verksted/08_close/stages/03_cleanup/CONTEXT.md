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
3. Run `verksted/_tools/wt-done.sh "$N" --pr <pr>`. With a worktree it refuses unless the worktree is clean, no process runs in it and the PR is `MERGED`; without one (the merge removed it) it needs the PR to be `MERGED`, deletes leftover branches and prints `already gone`. It removes the worktree, the local branch and the remote branch of this ticket only, then prunes. No `rm -rf`, no `--force`, no other path.
4. Why the worktree is often gone already: `gh pr merge --delete-branch` removes the worktree that has the PR head branch checked out. Verified: the gh 2.103.3 binary holds the messages `Removed worktree %s` and `Could not remove worktree %s; skipping local branch delete`, and `gh pr merge --help` says only "Delete the local and remote branch after merge". Not verified: whether a dirty worktree is removed, and the gh version that began it; no merge was run to test. Hence `clerk-check.sh` checks before the merge that the worktree is clean and at the PR head.
5. Write `report.md`: the paths removed and the final issue state.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
! git worktree list --porcelain | grep -q "worktrees/issue-$N-" && test -z "$(git branch --list "issue/$N-*")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
