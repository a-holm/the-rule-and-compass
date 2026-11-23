# 02_close: closing is its own act

One job: close the issue when every criterion row is met or owned by another open issue.

## Inputs

**This run:**
- the `verification.md` in the verifier record directory `OUT` (printed by `brief.sh verifier`)
- the `check.md` in `OUT` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../CONTEXT.md`

Do NOT load: closing through a commit or PR keyword.

## Process

1. If an input is missing or empty, stop and say which.
2. If the verdict is `Verdict: KEEP OPEN`, stop with `kept open: <the unmet and ownerless not run rows>`. List the unmet rows as new issues first if they are outside the original scope.
3. Every row reads `met`, or `not run (#M)` naming an open issue that owns the work (rule 58): `gh issue close "$N" --reason completed`. Otherwise stop with `kept open: <rows>`.

## Outputs

- `closed.md` -> `output/` (the `gh issue view` JSON, or the line `kept open: <rows>`)
- `check.md` -> `output/`

## Check

```sh
v="$OUT/verification.md"; grep -qx 'Verdict: CLOSE' "$v" && ! grep -qE '\| (unmet|not run) \|$' "$v" && for m in $(grep -oE 'not run \(#[0-9]+\)' "$v" | grep -oE '[0-9]+'); do test "$(gh issue view "$m" --json state -q .state)" = OPEN || exit 1; done && test "$(gh issue view "$N" --json state,stateReason -q '.state+" "+.stateReason')" = 'CLOSED COMPLETED'
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
