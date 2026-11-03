# 06_merge_cleanup: merge one at a time, clean path by path

One job: merge verified PRs into the base and remove only finished tickets' leftovers.

## Inputs

**This run:**
- `../05_receive_and_verify/output/verify.md`
- `../05_receive_and_verify/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../../_templates/brief-clerk.md`
- `../../../_tools/wt-done.sh`
- `../../CONTEXT.md`

Do NOT load: unfinished agents' worktrees, bulk deletion.

## Process

1. If an input is missing or empty, stop and say which.
2. The merge clerk (`brief.sh clerk`) merges one PR at a time into `BASE` only, never `main` (rule 41): `gh pr merge <pr> --squash --delete-branch --subject "<type(area): subject> (#<pr>)" --body-file <file>` with a hand-written subject and body. No gate per merge.
3. After each merge the clerk comments on the issue with the merge sha and the verified facts (rule 41). Then `git fetch` and update the integration worktree with `--ff-only`.
4. Clean up that ticket only with `verksted/_tools/wt-done.sh <issue>`. It refuses unless the worktree is clean, no process runs inside it and the PR is merged; then it removes the worktree, the local branch and the remote branch. Never `rm -rf`, never in bulk while agents live.
5. Write `report.md`: tickets merged with merge shas, tickets still open, deviations.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
awk -F'|' '/^\| #[0-9]/{gsub(/ /,"",$2);gsub(/ /,"",$3);print $2,$3}' "$V/03_orchestrate/stages/05_receive_and_verify/output/verify.md" | while read -r tk pr; do n=${tk#\#}; test "$(gh pr view "$pr" --json state -q .state)" = MERGED && ! git worktree list --porcelain | grep -q "worktrees/issue-$n-" && test -z "$(git branch --list "issue/$n-*")" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
