# 05_receive_and_verify: hand-backs checked against GitHub

One job: compare each agent's report with PR state, git and gate logs.

## Inputs

**This run:**
- `../03_dispatch/output/dispatch.md`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../../_tools/brief.sh`
- `../../../_tools/test-summary.sh`
- `../../../08_close/CONTEXT.md`
- `../../CONTEXT.md`

Do NOT load: the agents' prose, their transcripts.

## Process

1. If an input is missing or empty, stop and say which.
2. For each hand-back read the PR number and head sha it claims. Read the live values with `gh pr view <pr> --json headRefOid,state,baseRefName` and `git -C <worktree> rev-parse HEAD`.
3. Write `verify.md`: rows `| #N | <pr> | <sha from gh> | <sha from the report> |`. Re-read test counts with `verksted/_tools/test-summary.sh <log>...`; do not copy the agent's counts. A result with no log under `$GATES` counts as not run.
4. A mismatch goes back to the agent by SendMessage, with the number that differs.
5. Dispatch the verifier with `verksted/_tools/brief.sh verifier --issue <N> --pr <PR>` (model sonnet) and the claims file; it runs `../../../08_close/CONTEXT.md`. Never two verifiers at once on pull requests that touch the same files.

## Outputs

- `verify.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
awk -F'|' '/^\| #[0-9]/{gsub(/ /,"",$3);gsub(/ /,"",$4);gsub(/ /,"",$5);print $3,$4,$5}' "$V/03_orchestrate/stages/05_receive_and_verify/output/verify.md" | while read -r pr live said; do test "$live" = "$said" && test "$live" = "$(gh pr view "$pr" --json headRefOid -q .headRefOid)" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
