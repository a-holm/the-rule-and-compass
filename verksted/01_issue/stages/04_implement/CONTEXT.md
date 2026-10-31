# 04_implement: smallest change, targeted tests

One job: make the red proof green with the smallest change and commit it.

## Inputs

**This run:**
- `../03_prove_red/output/red.md`
- `../03_prove_red/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../../_shared/brief.md`
- `../../CONTEXT.md`

Do NOT load: the full suite, other tickets' branches, unrelated cleanups.

## Process

1. If an input is missing or empty, stop and say which.
2. Implement only what the run card names. Files outside it stay untouched.
3. Run only targeted tests (`--test <file> <filter>`) through the lock, `CARGO_BUILD_JOBS=2`, lock hold under 10 minutes. Log to `$G/issue$N-green.log`; a rerun overwrites it. Never start the full suite.
4. Commit in logical steps: `feat(area): ...` or `fix(area): ...`, body line `Refs #N`. Never `Closes`. No force-push, no rebase after a push: merge `BASE` instead.
5. No Python anywhere. Experiments are Rust.

## Outputs

- `implementation.md` -> `output/` (files changed, commands run, what was not verified)
- `check.md` -> `output/`

## Check

```sh
grep -q 'test result: ok' "$G/issue$N-green.log" && ! grep -q 'test result: FAILED' "$G/issue$N-green.log" && ! git -C "$W" log "$BASE"..HEAD --format=%B | grep -qiE '(close[sd]?|fix(es|ed)?|resolve[sd]?) #[0-9]+' && test -z "$(git -C "$W" status --porcelain)" && test -z "$(git -C "$W" ls-files '*.py')"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
