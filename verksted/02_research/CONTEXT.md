# 02_research: from a hard question to a binding decision

One job: turn an unclear or hard issue into a decision note that a Sonnet implementer follows, without changing code.

Evidence: #254, #257, #291, #292, #293, #352 and #364 each got a research pass before implementation. Wrong premises died at the measurement stage (#364: the locality test already passed). Research is a pipeline, not a stage of `../01_issue/CONTEXT.md`, because it uses a different model, writes no code and ends in a comment, not a PR.

## Inputs

**This run:**
- the issue number `N` and the `RUN` directory named by the dispatcher

**Reference (every run):**
- `../_shared/rules.md`
- `../_shared/commands.md`
- `../_shared/output-format.md`
- `../_templates/research-question.md`
- `../_templates/research-decision.md`
- `../../docs/time-series.md`

Do NOT load: other pipelines' stages, earlier research of other issues, the rest of `../_shared/`.

## Process

1. `stages/01_question/`: scope, overlapping issues, evidence files, numbered premises.
2. `stages/02_method/`: decompose with tractatus, read primary sources, tag every claim.
3. `stages/03_premise/`: measure each premise before anything is built.
4. `stages/04_decision/`: scope split, concrete spec, measurable acceptance criteria.
5. `stages/05_issue_comment/`: post the decision as one issue comment and write the report.

## Outputs

- `stages/05_issue_comment/output/report.md`: verdict, comment URL, premises that failed
- `stages/05_issue_comment/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
test "$(gh issue view "$N" --json state -q .state)" = OPEN && gh issue view "$N" --json comments -q '.comments[-1].body' | grep -q '^## Acceptance criteria'
```
