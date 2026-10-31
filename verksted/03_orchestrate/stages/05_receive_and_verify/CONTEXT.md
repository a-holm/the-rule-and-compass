# 05_receive_and_verify: hand-backs checked against GitHub

One job: compare each agent's report with PR state, git and gate logs.

## Inputs

**This run:**
- `../03_dispatch/output/dispatch.md`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: the agents' prose, their transcripts.

## Process

1. If an input is missing or empty, stop and say which.
2. For each hand-back read the PR number and head sha it claims. Read the live values with `gh pr view <pr> --json headRefOid,state,baseRefName` and `git -C <worktree> rev-parse HEAD`.
3. Write `verify.md`: rows `| #N | <pr> | <sha from gh> | <sha from the report> |`. Re-read test counts in the log files; do not copy the agent's counts.
4. A mismatch goes back to the agent by SendMessage, with the number that differs.

## Outputs

- `verify.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
awk -F'|' '/^\| #[0-9]/{gsub(/ /,"",$3);gsub(/ /,"",$4);gsub(/ /,"",$5);print $3,$4,$5}' "$V/03_orchestrate/stages/05_receive_and_verify/output/verify.md" | while read -r pr live said; do test "$live" = "$said" && test "$live" = "$(gh pr view "$pr" --json headRefOid -q .headRefOid)" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
