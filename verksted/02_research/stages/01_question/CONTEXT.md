# 01_question: scope, overlaps, premises

One job: state the question, the overlapping issues and the numbered premises to test.

## Inputs

**This run:**
- `gh issue view "$N" --json title,body,labels,comments`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_templates/research-question.md`
- `../../CONTEXT.md`

Do NOT load: code changes, other issues' research, the answer.

## Process

1. If the issue cannot be read, stop and say so.
2. Run on an Opus agent with tractatus thinking and context7: no code, no branch, no commit (`agent-model-policy`, `rules.md`).
3. Copy `_templates/research-question.md` to `output/question.md` and fill it. Name overlapping issues by number and the evidence files by path.
4. List the premises as `- P1: <claim that must be true>`. A premise is something a measurement can refute.

## Outputs

- `question.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
q="$V/02_research/stages/01_question/output/question.md"; for h in Scope Overlap Evidence Premises; do grep -q "^## $h" "$q" || exit 1; done; grep -q '^- P1: ' "$q" && grep -q '#[0-9]' "$q" && test -z "$(git -C "$RUN" diff HEAD --name-only)"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
