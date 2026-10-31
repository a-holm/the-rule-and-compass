# 03_full_gate: the one full gate

One job: run the CPU suite, the CUDA suite and the tiny benchmark once on the base tip.

## Inputs

**This run:**
- `../02_integration_merge/output/merged.md`
- `../02_integration_merge/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_tools/gate.sh`
- `../../CONTEXT.md`

Do NOT load: per-ticket reruns, anything but the three suites.

## Process

1. If an input is missing or empty, stop and say which.
2. Look first: no other cargo, rustc or `rcsr-bench` process (`ps -eo pcpu,args | grep -E 'cargo|rustc|rcsr-bench'`). Then run `verksted/_tools/gate.sh <integration worktree> <tag>` as one background shell and wait for its notification.
3. Write `gate.md` from the output: `cpu: ok=N failed=0`, `cuda: ok=N failed=0`, the bench line, `tested: <sha>`, and the three log paths.
4. A red suite is not retried silently. A flake gets its own issue (`../../_shared/failure-modes.md`).

## Outputs

- `gate.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
g="$V/04_gate/stages/03_full_gate/output/gate.md"; grep -q '^cpu: ok=[0-9]* failed=0$' "$g" && grep -q '^cuda: ok=[0-9]* failed=0$' "$g" && grep -q 'recovered 27/27' "$g" && grep -q '^tested: [0-9a-f]\{40\}$' "$g"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
