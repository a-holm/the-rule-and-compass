# 03_cleanup: path by path, only mine

One job: remove this ticket's worktree and branches and the verifier worktree, and nothing else.

## Inputs

**This run:**
- `../02_close/output/closed.md` (the closed issue, or the line `kept open: <rows>`: the pull request is merged either way)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`
- `../../../_tools/wt-done.sh`
- `../../../_tools/verify-merged.sh`

Do NOT load: other agents' worktrees, bulk deletion, rm -rf.

## Process

1. If an input is missing or empty, stop and say which.
2. Confirm the ticket's agent is finished: its PR is `MERGED` and no process runs under its worktree. An orchestrator brief that calls the author worktree read only means read only until the merge; this stage still applies.
3. Run `verksted/_tools/wt-done.sh "$N" --pr <pr>`. With a worktree it refuses unless the worktree is clean, no process runs in it and the PR is `MERGED`; without one (the merge removed it) it needs the PR to be `MERGED`, deletes leftover branches and prints `already gone`. It removes the worktree, the local branch and the remote branch of this ticket only, then prunes. No `rm -rf`, no `--force`, no other path. Then run `verksted/_tools/verify-merged.sh "$PR" --done`: it aborts an unfinished merge and removes `$WT/verifier-$PR` (rule 59).
4. Why the worktree is often gone already: the merge with `--delete-branch` removes the worktree that has the PR head branch checked out. Verified: the gh 2.103.3 binary holds the messages `Removed worktree %s` and `Could not remove worktree %s; skipping local branch delete`, and the merge help of that binary says only "Delete the local and remote branch after merge". Not verified: whether a dirty worktree is removed, and the gh version that began it; no merge was run to test. Hence `verifier-check.sh` checks before the merge that the worktree is clean and at the PR head.
5. Write `report.md`: the paths removed and the final issue state (`closed` or `kept open: <rows>`).

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
! git worktree list --porcelain | grep -q "worktrees/issue-$N-" && test ! -e "$WT/verifier-$PR" && test -z "$(git branch --list "issue/$N-*")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
