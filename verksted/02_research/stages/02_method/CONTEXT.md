# 02_method: claims with a falsifier

One job: decompose the question and gather primary sources, tagging every claim.

## Inputs

**This run:**
- `../01_question/output/question.md`
- `../01_question/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/failure-modes.md`
- `../../CONTEXT.md`

Do NOT load: implementation, the premise measurements, the decision.

## Process

1. If an input is missing or empty, stop and say which.
2. Decompose with `mcp__tractatus_thinking__tractatus_thinking`. Read primary sources: papers, vendor docs, the repo. Use context7 for library semantics. Web search for the rest.
3. Write `method.md`. Every claim is one bullet: `- [measured|derived|inferred] <claim> | falsifier: <test that would refute it>`.
4. For a hang or deadlock, add the wait-for graph and list every cycle.

## Outputs

- `method.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
m="$V/02_research/stages/02_method/output/method.md"; test "$(grep -c '^- \[' "$m")" -ge 1 && test "$(grep '^- \[' "$m" | grep -vcE '^- \[(measured|derived|inferred)\] .+ \| falsifier: .+')" = 0
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
