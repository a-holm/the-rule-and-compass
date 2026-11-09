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
- `../../../_tools/rc-run.sh`
- `../../../_tools/sanitize.sh`
- `../../../_tools/test-summary.sh`
- `../../../_tools/wait-for.sh`
- `../../../_tools/ab-pair.sh`
- `../../../_tools/bin-pin.sh`

Do NOT load: the full suite, other tickets' branches, unrelated cleanups.

## Process

1. If an input is missing or empty, stop and say which.
2. Implement only what the run card names. Files outside it stay untouched.
3. Build and test only through `verksted/_tools/rc-run.sh` (`build`, `test`; a CPU-only test omits `--cuda` and uses the slot). Run only targeted tests (`--test <file> <filter>`), never the full suite. Log to `$GATES/issue$N-green.log`; a rerun overwrites it.
4. Wait with one background shell, or `verksted/_tools/wait-for.sh --log <log>`. Leave no background job after the report (rule 37).
5. GPU: any `.cu` change, allocation-failure path or device test of changed code runs first with `verksted/_tools/sanitize.sh`. Search-level tests set `DispatchMode::Device`.
6. The build log shows `warnings: 0` in `test-summary.sh` output. Run `--test test_attributes` and `--test sanitizer_plan` once at the end.
7. Keep hunks small and local in `src/search.rs`, `src/main.rs`, `src/schema.rs` and `tests/gpu_fit.rs`: other agents edit them in parallel.
8. A new refusal or check on a shared path: grep `tests/` for every test of that path and run those files too (rule 36).
9. A performance claim is measured with `verksted/_tools/ab-pair.sh` on two binaries pinned by `verksted/_tools/bin-pin.sh` (rule 38). The numbers go in the issue.
10. Commit in logical steps: `feat(area): ...` or `fix(area): ...`, body line `Refs #N`. Never `Closes`. No force-push, no rebase after a push: merge `BASE` instead.
11. No Python anywhere. Experiments are Rust.

## Outputs

- `implementation.md` -> `output/` (files changed, commands run, what was not verified)
- `check.md` -> `output/`

## Check

```sh
verksted/_tools/test-summary.sh "$GATES/issue$N-green.log" >/dev/null && ! git -C "$W" log "$BASE"..HEAD --format=%B | grep -qiE '(close[sd]?|fix(es|ed)?|resolve[sd]?) #[0-9]+' && test -z "$(git -C "$W" status --porcelain)" && test -z "$(git -C "$W" ls-files '*.py')"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
