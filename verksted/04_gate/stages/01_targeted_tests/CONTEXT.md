# 01_targeted_tests: each ticket tested narrowly

One job: collect the targeted-test evidence of every ticket in the gate.

## Inputs

**This run:**
- `gh pr list --base "$BASE" --state merged`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: the full suite, other tickets' sources.

## Process

1. List the tickets in this gate in `tickets.md`: one row `| #N | <log path> |` each.
2. Every log lives in `$G/issue<N>-*.log` and shows `test result: ok`. A ticket without a green targeted log is not in the gate.
3. Each ticket's PR body lists what it did not verify. Copy that line into `tickets.md`.

## Outputs

- `tickets.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
t="$V/04_gate/stages/01_targeted_tests/output/tickets.md"; test "$(grep -c '^| #[0-9]' "$t")" -ge 1 && for n in $(grep -o '^| #[0-9]*' "$t" | tr -d '| #'); do ls "$G"/issue"$n"-*.log > /dev/null 2>&1 && grep -lq 'test result: ok' "$G"/issue"$n"-*.log || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
