# 04_gate: one full gate on the integration tip

One job: prove that the integration branch builds and passes, once for all merged tickets.

Evidence: 20 distinct full gates in `$GATES` for a handful of PRs, each about an hour of CPU and GPU; the user's correction on 2026-10-30 ("Bare merge dem til integration worktree ... kjøre gaten en gang til slutt"). Agents run targeted tests only; this pipeline owns the single full gate.

## Inputs

**This run:**
- the integration branch `BASE` and the list of merged tickets

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/output-format.md`
- `../_tools/gate.sh`
- `../_tools/test-summary.sh`
- `../../AGENTS.md`

Do NOT load: the ticket pipelines' stages, earlier gate logs, the rest of `../_shared/`.

## Process

1. [01_targeted_tests](stages/01_targeted_tests/CONTEXT.md): collect each ticket's targeted-test evidence.
2. [02_integration_merge](stages/02_integration_merge/CONTEXT.md): confirm every ticket is in `BASE`, new tickets branch from it.
3. [03_full_gate](stages/03_full_gate/CONTEXT.md): run `verksted/_tools/gate.sh` once, after all issues are done.
4. [04_tested_vs_merged](stages/04_tested_vs_merged/CONTEXT.md): prove the tested tree is the tree that is merged.
5. [05_merge_main](stages/05_merge_main/CONTEXT.md): only on the project lead's written instruction: merge the integration branch into `main`, then write the report. Without it the stage fails.

## Outputs

- `stages/05_merge_main/output/report.md`: gate numbers, tested sha, merge sha
- `stages/05_merge_main/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/04_gate/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
