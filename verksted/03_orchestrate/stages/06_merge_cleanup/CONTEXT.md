# 06_merge_cleanup: after each verifier run

One job: bring the integration worktree up to date after each verifier run and report what merged.

## Inputs

**This run:**
- `../05_receive_and_verify/output/verify.md`
- `../05_receive_and_verify/output/check.md` (must be `pass`)
- the `report.md` of each verifier run (`../../../08_close/stages/03_cleanup/output/report.md` and the `report.md` in the verifier record directory `OUT`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: unfinished agents' worktrees, bulk deletion.

## Process

1. If an input is missing or empty, stop and say which.
2. Read the verifier's `report.md`. The merge, the comment on the issue, the close and the clean-up of the ticket's worktree and branches are `../../../08_close/CONTEXT.md` (one home): this stage runs none of them.
3. `git fetch` and update the integration worktree with `--ff-only`.
4. Write `report.md`: tickets merged with merge shas, the line `kept open: #N #M` (or `kept open: none`), tickets whose tip checks were not clean, deviations. A ticket on the `kept open:` line is exempt from the merged-PR and clean-up test; every other row of `verify.md` must pass it. An empty `verify.md` or `report.md` fails.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
v="$V/03_orchestrate/stages/05_receive_and_verify/output/verify.md"; r="$V/03_orchestrate/stages/06_merge_cleanup/output/report.md"; rows=$(awk -F'|' '/^\| #[0-9]/{gsub(/ /,"",$2);gsub(/ /,"",$3);print $2,$3}' "$v"); test -n "$rows" && test -s "$r" && printf '%s\n' "$rows" | while read -r tk pr; do n=${tk#\#}; if sed -n 's/^kept open: //p' "$r" | grep -qE "(^| )#$n( |\$)"; then continue; fi; test "$(gh pr view "$pr" --json state -q .state)" = MERGED && ! git worktree list --porcelain | grep -q "worktrees/issue-$n-" && test -z "$(git branch --list "issue/$n-*")" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
