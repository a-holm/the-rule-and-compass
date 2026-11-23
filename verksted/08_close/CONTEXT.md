# 08_close: from a pull request to a merged, verified and closed issue and a clean checkout

One job: verify a pull request on the tree that merges, merge it, close its issue, and remove the leftovers path by path.

Evidence: GitHub closed issues through `Closes` in pushed commits; issues were closed without every part implemented ("completely ridiculous to close an issue without implementing every single part", 2026-10-20); the #257 verifier found three gaps behind a green PR (inferred, not checked) and the issue closed only after PR #348. A merged issue is verified at the merge (merged means verified, user, 2026-11-22), so one routine checks, merges and verifies. The closure gate is `../../docs/issue-workflow.md` section 5.

## Inputs

**This run:**
- the issue number `N`, the pull request number `PR` and the hand-back claims

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/output-format.md`
- `../_templates/verification-comment.md`
- `../_templates/merge-body.md`
- `../../docs/issue-workflow.md`
- `../_tools/verify-merged.sh`
- `../_tools/wt-done.sh`
- `../_tools/test-summary.sh`

Do NOT load: the implementation pipelines' stages, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. `stages/01_verifier/`: the verifier (rule 69) checks evidence and criteria on the merged tree, merges, checks the tip, ticks the boxes it verified and posts one comment.
2. `stages/02_close/`: close the issue as a separate act, when every row is `met` or `not run (#M)` with an open issue `#M` that owns the work (rule 58).
3. `stages/03_cleanup/`: remove the worktree, local branch and remote branch of this ticket and the verifier worktree.

## Outputs

- `stages/03_cleanup/output/report.md`: verdict, issue state, paths removed
- `stages/03_cleanup/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
r="$V/08_close/stages/03_cleanup/output/report.md"; { test "$(gh issue view "$N" --json state -q .state)" = CLOSED || grep -q '^kept open: ' "$r"; } && test ! -e "$W" && test ! -e "$WT/verifier-$PR" && test -z "$(git ls-remote --heads origin "issue/$N-$SLUG")"
```
