# 01_verifier: every criterion has evidence

One job: map each acceptance criterion to named evidence and post the verdict.

## Inputs

**This run:**
- `gh issue view "$N" --json body,comments`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_templates/verification-comment.md`
- `../../CONTEXT.md`

Do NOT load: running cargo, fixing code, the implementer's prose.

## Process

1. The verifier is not the implementer. It runs no cargo: it reads code, the merged diff, PR bodies and gate logs.
2. Copy `_templates/verification-comment.md` to `output/verification.md`. One row per acceptance criterion: `| AC1 | <criterion> | <test name or log path> | met|unmet |`.
3. Check the six closure conditions of `issue-workflow.md` section 5. An inferred claim is not evidence.
4. End with `Verdict: CLOSE` or `Verdict: KEEP OPEN`. Post it once with `gh issue comment "$N" --body-file`.

## Outputs

- `verification.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
v="$V/08_close/stages/01_verifier/output/verification.md"; ac=$(gh issue view "$N" --json body -q .body | sed -n '/^## Acceptance criteria/,/^## /p' | grep -c '^- \[[ x]\]'); test "$ac" -ge 1 && test "$(grep -c '^| AC[0-9]' "$v")" = "$ac" && grep -qE '^Verdict: (CLOSE|KEEP OPEN)$' "$v" && test "$(gh issue view "$N" --json comments -q '.comments[-1].body')" = "$(cat "$v")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
