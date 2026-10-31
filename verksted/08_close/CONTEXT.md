# 08_close: from merged code to a closed issue and a clean checkout

One job: close an issue only after an independent verifier maps every acceptance criterion to evidence, then remove its leftovers path by path.

Evidence: GitHub closed issues through `Closes` in pushed commits; issues were closed without every part implemented ("completely ridiculous to close an issue without implementing every single part", 2026-10-20); the #257 verifier found three gaps behind a green PR (inferred, not checked) and the issue closed only after PR #348. The closure gate is `../../docs/issue-workflow.md` section 5.

## Inputs

**This run:**
- the issue number `N`, its merged PR numbers and the integration gate report

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/output-format.md`
- `../_templates/verification-comment.md`
- `../../docs/issue-workflow.md`

Do NOT load: the implementation pipelines' stages, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. `stages/01_verifier/`: an independent Sonnet verifier maps each criterion to evidence and posts one comment.
2. `stages/02_close/`: close the issue as a separate act, only on a CLOSE verdict.
3. `stages/03_cleanup/`: remove the worktree, local branch and remote branch of this ticket only.

## Outputs

- `stages/03_cleanup/output/report.md`: verdict, issue state, paths removed
- `stages/03_cleanup/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
test "$(gh issue view "$N" --json state -q .state)" = CLOSED && test ! -e "$W" && test -z "$(git ls-remote --heads origin "issue/$N-$SLUG")"
```
