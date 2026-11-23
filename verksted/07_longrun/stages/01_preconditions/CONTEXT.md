# 01_preconditions: start only when nothing is left

One job: prove that the run may start now.

## Inputs

**This run:**
- `gh issue list --label roadmap --state open --limit 200`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`
- `../../../_tools/bin-pin.sh`
- `../../../_tools/run-record.sh`

Do NOT load: launching, tuning flags.

## Process

1. Issues that block the run are the ones the user names. List them in `precond.md` as `blocking: #N #M` (or `blocking: none`); each must be closed. Other open roadmap issues, parked or not, do not block.
2. The binary comes from the tip of `$BASE` (the integration branch): `$RUN` is a worktree whose `git rev-parse HEAD` equals `origin/$BASE`, built in the build slot. Pin it with `../../../_tools/bin-pin.sh <worktree> <run>/bin`. Check: `sha256sum` of `<run>/bin/rcsr` equals `<run>/bin/rcsr.sha256`, and `<run>/bin/rcsr --version` exits 0 (a pinned binary that needs its build directory failed with exit 127).
3. Open the run card once the binary is pinned: `../../../_tools/run-record.sh new --id "$(date -u +%Y%m%d)-$LABEL" --kind <kind> --bin "$RUN/bin/rcsr" --worktree "$RUN" --store <store>...` (one `--store` per store the run reads; `--mask` and `--seed` when known). The card lives in the local area (`../../../_shared/local.md`); `precond.md` names its path. A second run with the same label and day needs a new slug.
4. Nothing of an earlier run of this `$RUN` runs: every pid in `$RUN/run.pid` (one per line, written by `../02_launch`) is gone (`kill -0` fails). Processes of other runs and campaign units (rules 65 and 66) do not count. Never `pgrep -f`.
5. Once the checks pass, start at once. Do not wait idle.

## Outputs

- `precond.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
p="$V/07_longrun/stages/01_preconditions/output/precond.md"; grep -q '^blocking: ' "$p" || exit 1; for n in $(sed -n 's/^blocking: //p' "$p" | grep -oE '[0-9]+'); do test "$(gh issue view "$n" --json state -q .state)" = CLOSED || exit 1; done; if test -s "$RUN/run.pid"; then for q in $(cat "$RUN/run.pid"); do ! kill -0 "$q" 2> /dev/null || exit 1; done; fi; test "$(git -C "$RUN" rev-parse HEAD)" = "$(git -C "$ROOT" rev-parse "origin/$BASE")" && (cd "$RUN/bin" && sha256sum -c rcsr.sha256) && "$RUN/bin/rcsr" --version > /dev/null && "$V/_tools/run-record.sh" latest --slug "$LABEL" > /dev/null
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
