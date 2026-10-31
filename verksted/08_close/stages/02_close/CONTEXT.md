# 02_close: closing is its own act

One job: close the issue after a CLOSE verdict.

## Inputs

**This run:**
- `../01_verifier/output/verification.md`
- `../01_verifier/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/rules.md`
- `../../CONTEXT.md`

Do NOT load: closing through a commit or PR keyword.

## Process

1. If an input is missing or empty, stop and say which.
2. If the verdict is `Verdict: KEEP OPEN`, stop with `fail: kept open`. List the unmet rows as new issues first if they are outside the original scope.
3. Every row must read `met`. `gh issue close "$N" --reason completed`.

## Outputs

- `closed.md` -> `output/` (the `gh issue view` JSON)
- `check.md` -> `output/`

## Check

```sh
grep -qx 'Verdict: CLOSE' "$V/08_close/stages/01_verifier/output/verification.md" && ! grep -q '| unmet |' "$V/08_close/stages/01_verifier/output/verification.md" && test "$(gh issue view "$N" --json state,stateReason -q '.state+" "+.stateReason')" = 'CLOSED COMPLETED'
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
