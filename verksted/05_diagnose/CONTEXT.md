# 05_diagnose: from a stuck or diverging GPU run to a classified cause

One job: find out why a GPU test or run hangs or disagrees with the CPU, prove it small, fix it with bounded waits, and classify every parity delta.

Evidence: #303 held the lock and 11 GB of GPU for over 90 minutes (I/O counter frozen, 0.6 % CPU, all threads in `futex_do_wait`); #352 found the projected-ring deadlock (12 of 12 repro hangs); #353 made processes attachable; #355, #357, #359 and #364 sorted parity deltas into production bugs and numeric tiers. A drive-by fix of a bug found by the harness is not allowed: it gets its own issue.

## Inputs

**This run:**
- the stuck or diverging command, its log and the pid, if any

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/failure-modes.md`
- `../_shared/output-format.md`
- `../../docs/issue-workflow.md`
- `../../scripts/stacks.sh`
- `../../scripts/stall-watch.sh`

Do NOT load: the data and long-run pipelines, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. `stages/01_prove_stuck/`: numbers that separate stuck from slow.
2. `stages/02_stack/`: backtraces of every thread.
3. `stages/03_wait_graph/`: the wait-for graph and every cycle.
4. `stages/04_repro_red/`: a small deterministic repro that hangs on unfixed code.
5. `stages/05_fix_bounded/`: the fix, plus bounded waits so a future bug fails loudly.
6. `stages/06_medium_run/`: a medium-store run under 10 minutes instead of the huge test.
7. `stages/07_classify_deltas/`: production bug (own issue) or numeric tier (declared).

## Outputs

- `stages/07_classify_deltas/output/report.md`: cause, fix, deltas with their issue or tier
- `stages/07_classify_deltas/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/05_diagnose/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
