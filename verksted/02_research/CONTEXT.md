# 02_research: from a hard question to a binding decision

One job: turn an unclear or hard issue into a decision note that a Sonnet implementer follows, without changing code.

Model policy: [`models.md`](../_shared/models.md). Research plans, so it runs on Opus, or on Fable 5.2 for an extraordinarily hard case, with tractatus thinking, context7 and web search whenever it helps. Implementation and verification use Sonnet (rule 40).

Evidence: #254, #257, #291, #292, #293, #352 and #364 each got a research pass before implementation. Wrong premises died at the measurement stage (#364: the locality test already passed). Research is a pipeline, not a stage of `../01_issue/CONTEXT.md`, because it uses a different model, writes no code and ends in a comment, not a PR.

## Inputs

**This run:**
- the issue number `N` and the `RUN` directory named by the dispatcher

**Reference (every run):**
- `../_shared/rules.md`
- `../_shared/commands.md`
- `../_tools/wt-new.sh`
- `../_tools/wt-done.sh`
- `../_tools/rc-run.sh`
- `../_tools/ab-pair.sh`
- `../_tools/issue-lint.sh`
- `../_shared/output-format.md`
- `../_templates/research-question.md`
- `../_templates/research-decision.md`
- `../../docs/time-series.md`

Do NOT load: other pipelines' stages, earlier research of other issues, the rest of `../_shared/`.

## Process

1. [01_question](stages/01_question/CONTEXT.md): scope, overlapping issues, evidence files, numbered premises.
2. [02_method](stages/02_method/CONTEXT.md): decompose with tractatus, read primary sources, tag every claim.
3. [03_premise](stages/03_premise/CONTEXT.md): measure each premise before anything is built.
4. [04_decision](stages/04_decision/CONTEXT.md): scope split, concrete spec, measurable acceptance criteria.
5. [05_issue_comment](stages/05_issue_comment/CONTEXT.md): post the decision as one issue comment and write the report.

## Outputs

- `stages/05_issue_comment/output/report.md`: verdict, comment URL, premises that failed
- `stages/05_issue_comment/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
test "$(gh issue view "$N" --json state -q .state)" = OPEN && gh issue view "$N" --json comments -q '.comments[-1].body' | grep -q '^## Acceptance criteria'
```
