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
- `../../../_tools/rc-run.sh`
- `../../../_tools/test-summary.sh`

Do NOT load: the fix, docs, the PR.

## Process

1. If an input is missing or empty, stop and say which.
2. Write the failing test or repro first (Rust only, no Python). When the fix already exists, show red on a temporary WIP commit that reverts the fix. Never use `git stash` (shared by all worktrees). Drop the WIP commit before the first push.
3. Build: `verksted/_tools/rc-run.sh build "$W" "$N" red-build [--cuda] -- cargo test --release [--features cuda] --no-run --test <file>`. Run: `verksted/_tools/rc-run.sh test "$W" "$N" red [--cuda] -- cargo test --release [--features cuda] --test <file> <filter>`. The log is `$GATES/issue$N-red.log`.
4. `verksted/_tools/test-summary.sh` on that log must print `tests: passed=<n> failed=<k>` with k at least 1 and `errors: 0`. A compile error is not red. The `error: test failed` line cargo prints after a red test is expected.
5. A documentation-only issue (`area:docs`, no behaviour) writes `n/a: <reason>` as the first line of `red.md` and names its own red proof, for example a deliberately broken link.
6. Write `red.md`: command, log path, failing test names.

## Outputs

- `red.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
s=$(verksted/_tools/test-summary.sh "$GATES/issue$N-red.log"); { echo "$s" | grep -qE '^tests: passed=[0-9]+ failed=[1-9]' && echo "$s" | grep -qx 'errors: 0'; } || head -1 "$V/01_issue/stages/03_prove_red/output/red.md" | grep -q '^n/a: '
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
