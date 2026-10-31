# 06_merge_cleanup: merge one at a time, clean path by path

One job: merge verified PRs into the base and remove only finished tickets' leftovers.

## Inputs

**This run:**
- `../05_receive_and_verify/output/verify.md`
- `../05_receive_and_verify/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: unfinished agents' worktrees, bulk deletion.

## Process

1. If an input is missing or empty, stop and say which.
2. Merge one PR at a time into `BASE` with `gh pr merge <pr> --squash` and a hand-written subject and body (section "Merge" of `commands.md`). No gate per merge. Do not use `--delete-branch` while the worktree exists.
3. After each merge: `git fetch`, update the integration worktree with `--ff-only`, then clean that ticket only: `git worktree remove`, `git branch -D`, `git push origin --delete`, `git worktree prune`. Never `rm -rf`, never in bulk while agents live.
4. Write `report.md`: tickets merged with merge shas, tickets still open, deviations.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
awk -F'|' '/^\| #[0-9]/{gsub(/ /,"",$2);gsub(/ /,"",$3);print $2,$3}' "$V/03_orchestrate/stages/05_receive_and_verify/output/verify.md" | while read -r tk pr; do n=${tk#\#}; test "$(gh pr view "$pr" --json state -q .state)" = MERGED && ! git worktree list --porcelain | grep -q "worktrees/issue-$n-" && test -z "$(git branch --list "issue/$n-*")" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
