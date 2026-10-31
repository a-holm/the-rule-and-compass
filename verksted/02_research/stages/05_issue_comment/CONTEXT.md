# 05_issue_comment: post the decision once

One job: post the decision as one issue comment and write the report.

## Inputs

**This run:**
- `../04_decision/output/decision.md`
- `../04_decision/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: closing or relabelling the issue, implementation.

## Process

1. If an input is missing or empty, stop and say which.
2. `gh issue comment "$N" --body-file <decision.md>`. One comment. Do not close, relabel or reassign the issue.
3. Write `report.md`: comment URL, premises that failed, the line the implementer brief must quote ("the decision in <URL> is the binding design").

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test "$(gh issue view "$N" --json state -q .state)" = OPEN && test "$(gh issue view "$N" --json comments -q '.comments[-1].body')" = "$(cat "$V/02_research/stages/04_decision/output/decision.md")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
