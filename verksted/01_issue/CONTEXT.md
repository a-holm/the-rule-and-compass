# 01_issue: from open issue to reviewed pull request

One job: take one issue to an open pull request, red first, with docs in sync and the issue still open.

Evidence: every ticket since #254 followed this shape (docs/issue-workflow.md, 39 agent briefs, PRs up to #369). Gate, merge and close are separate pipelines (`../04_gate/CONTEXT.md`, `../08_close/CONTEXT.md`) because they run once for many tickets.

## Inputs

**This run:**
- the issue number `N`, the `SLUG` and the `BASE` branch from the dispatcher

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/output-format.md`
- `../_templates/run.md`
- `../_templates/pr.md`
- `../../AGENTS.md`
- `../../docs/issue-workflow.md`

Do NOT load: other pipelines' stages, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. `stages/01_worktree/`: create `worktrees/issue-N-SLUG` on `issue/N-SLUG` from `BASE`.
2. `stages/02_issue_body/`: freeze the issue into a run card, one proof line per acceptance criterion.
3. `stages/03_prove_red/`: a failing test or repro on the unfixed code.
4. `stages/04_implement/`: smallest change, targeted tests under the lock, commits with `Refs #N`.
5. `stages/05_docs_sync/`: `--help`, `rcsr schema`, README and `docs/` match the change.
6. `stages/06_pr/`: push, open the pull request against `BASE`, write the report.

## Outputs

- `stages/06_pr/output/report.md`: verdict, evidence, deviations
- `stages/06_pr/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
pr=$(gh pr view "issue/$N-$SLUG" --json state,headRefOid -q '.state+" "+.headRefOid'); test "$pr" = "OPEN $(git -C "$W" rev-parse HEAD)" && test -z "$(git -C "$W" status --porcelain)" && test "$(gh issue view "$N" --json state -q .state)" = OPEN
```
