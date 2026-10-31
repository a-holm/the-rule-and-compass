# 06_pr: push and open the pull request

One job: push the branch, open the PR against the base and write the report.

## Inputs

**This run:**
- `../05_docs_sync/output/docs.md`
- `../05_docs_sync/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_templates/pr.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: merge, gate, close.

## Process

1. If an input is missing or empty, stop and say which.
2. `git push -u origin "issue/$N-$SLUG"`, then `gh pr create --base "$BASE" --body-file <filled pr.md>`. Do not merge.
3. The PR body lists every acceptance criterion with its box ticked or open, commands with counts, deviations, and `Refs #N`.
4. Write `report.md` per `output-format.md`: PR number, head sha, criteria with evidence, what was not verified.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
b=$(gh pr view "issue/$N-$SLUG" --json body -q .body); test "$(gh pr view "issue/$N-$SLUG" --json baseRefName -q .baseRefName)" = "$BASE" && echo "$b" | grep -q "Refs #$N" && ! echo "$b" | grep -qiE '(closes|fixes|resolves) #' && for h in 'What changed' 'Acceptance criteria' 'Verification' 'Deviations'; do echo "$b" | grep -q "^## $h" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
