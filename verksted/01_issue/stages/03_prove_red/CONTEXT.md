# 03_prove_red: failing test before the fix

One job: show on unfixed code that each testable criterion fails.

## Inputs

**This run:**
- `../02_issue_body/output/run.md`
- `../02_issue_body/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../../_shared/failure-modes.md`
- `../../CONTEXT.md`

Do NOT load: the fix, docs, the PR.

## Process

1. If an input is missing or empty, stop and say which.
2. Write the failing test or repro first (Rust only, no Python). Run it through the lock, narrow, with an outer `timeout`; log to `$G/issue$N-red.log`.
3. The log must show `test result: FAILED` for the new test and no `error[E....]` or `could not compile` line. A compile error is not red. The `error: test failed` line cargo prints after a red test is expected.
4. A documentation-only issue (`area:docs`, no behaviour) writes `n/a: <reason>` as the first line of `red.md` and names its own red proof, for example a deliberately broken link.
5. Write `red.md`: command, log path, failing test names.

## Outputs

- `red.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
grep -q 'test result: FAILED' "$G/issue$N-red.log" && ! grep -qE 'could not compile|^error\[E[0-9]+\]' "$G/issue$N-red.log" || head -1 "$V/01_issue/stages/03_prove_red/output/red.md" | grep -q '^n/a: '
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
