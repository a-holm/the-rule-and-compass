# 03_full_gate: the one full gate

One job: run the CPU suite, the CUDA suite and the tiny benchmark once on the base tip.

## Inputs

**This run:**
- `../02_integration_merge/output/merged.md`
- `../02_integration_merge/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_tools/gate.sh`
- `../../../_tools/test-summary.sh`
- `../../../_tools/sanitize-all.sh`
- `../../../../my-version/tests/sanitizer.plan`
- `../../CONTEXT.md`

Do NOT load: per-ticket reruns, anything but the three suites.

## Process

1. If an input is missing or empty, stop and say which.
2. Look first: no other cargo, rustc or `rcsr-bench` process (`ps -eo pcpu,args | grep -E 'cargo|rustc|rcsr-bench'`). Then run `verksted/_tools/gate.sh <integration worktree> <tag>` as one background shell and wait for its notification.
3. Write `gate.md` from the output: `cpu: ok=N failed=0`, `cuda: ok=N failed=0`, the bench line, `tested: <sha>`, and the three log paths.
4. The memcheck pass is `verksted/_tools/sanitize-all.sh <integration worktree> <tag>` as one background shell. It runs every CUDA test binary from `my-version/tests/sanitizer.plan` under compute-sanitizer through the lock, one hold per run of at most 580 s, and writes `<tag>-san-summary.log`. `gate.sh` has no sanitizer stage: this step, not the script, calls `sanitize-all.sh`. A `skip`, `api-errors-no`, `no-cuda`, `no-cuda-app-only` or `run-bin` line in the plan carries its reason (a `run-bin` line carries the arguments of the binary, for example `rcsr-gpu-bench * run-bin --batch 64 --reps 1 --depth 3`); a binary that exits 255 without a CUDA call and has no `no-cuda` line fails as `unexpected no CUDA call`. Exit 0 needs every run at rc 0 with `ERROR SUMMARY: 0 errors`; a `budget` line means a run hit the 580 s timeout. Do not raise the timeout; size the fixture or add a plan line with a reason.
5. A red suite is not retried silently. A flake gets its own issue (`../../_shared/failure-modes.md`).

## Outputs

- `gate.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
g="$V/04_gate/stages/03_full_gate/output/gate.md"; grep -q '^cpu: ok=[0-9]* failed=0$' "$g" && grep -q '^cuda: ok=[0-9]* failed=0$' "$g" && grep -q 'recovered 27/27' "$g" && grep -q '^tested: [0-9a-f]\{40\}$' "$g"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
