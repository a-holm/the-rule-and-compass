# 04_gate: one full gate on the integration tip

One job: prove that the integration branch builds and passes, once for all merged tickets, and land it on `main`.

Evidence: 20 distinct full gates in `/mnt/lager/tmp/eureqa-gates/` for a handful of PRs, each about an hour of CPU and GPU; the user's correction on 2026-10-30 ("Bare merge dem til integration worktree ... kjøre gaten en gang til slutt"). Agents run targeted tests only; this pipeline owns the single full gate.

## Inputs

**This run:**
- the integration branch `BASE` and the list of merged tickets

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/output-format.md`
- `../_tools/gate.sh`
- `../../AGENTS.md`

Do NOT load: the ticket pipelines' stages, earlier gate logs, the rest of `../_shared/`.

## Process

1. `stages/01_targeted_tests/`: collect each ticket's targeted-test evidence.
2. `stages/02_integration_merge/`: confirm every ticket is in `BASE`, new tickets branch from it.
3. `stages/03_full_gate/`: run `verksted/_tools/gate.sh` once.
4. `stages/04_tested_vs_merged/`: prove the tested tree is the tree that is merged.
5. `stages/05_merge_main/`: merge the integration branch into `main`, then write the report.

## Outputs

- `stages/05_merge_main/output/report.md`: gate numbers, tested sha, merge sha
- `stages/05_merge_main/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/04_gate/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
