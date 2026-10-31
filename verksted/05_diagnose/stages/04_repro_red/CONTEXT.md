# 04_repro_red: small, deterministic, red

One job: reproduce the hang in a small test that fails on the unfixed code.

## Inputs

**This run:**
- `../03_wait_graph/output/graph.md`
- `../03_wait_graph/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: the fix, the huge original test.

## Process

1. If an input is missing or empty, stop and say which.
2. Write a Rust test or example that hangs on the unfixed commit. Run it N times under an outer `timeout -s KILL 300`, through the lock. For a deeper look run it under `gdb -q -batch -nx -ex run -ex 'thread apply all bt 25' --args <bin> <test> --exact --nocapture --test-threads=1`.
3. Write `repro.md`: the commit, the command, the log path and the line `hangs: K/N`.

## Outputs

- `repro.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
r="$V/05_diagnose/stages/04_repro_red/output/repro.md"; k=$(sed -n 's|^hangs: \([0-9]*\)/\([0-9]*\)$|\1 \2|p' "$r"); set -- $k; test "$#" = 2 && test "$1" -ge 1 && test "$2" -ge 5
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
