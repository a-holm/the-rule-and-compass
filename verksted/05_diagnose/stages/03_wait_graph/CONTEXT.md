# 03_wait_graph: every wait cycle on paper

One job: build the wait-for graph and list every cycle before touching code.

## Inputs

**This run:**
- `../02_stack/output/stacks.txt`
- `../02_stack/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/failure-modes.md`
- `../../CONTEXT.md`

Do NOT load: the fix, the repro.

## Process

1. If an input is missing or empty, stop and say which.
2. Decompose with `mcp__tractatus_thinking__tractatus_thinking`. One line per edge in `graph.md`: `- <thread or lock A> waits for <B> at <file:line>`.
3. List each cycle as `cycle: A -> B -> A`. If there is none, write `no cycle: <what the threads wait for instead>`.

## Outputs

- `graph.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
g="$V/05_diagnose/stages/03_wait_graph/output/graph.md"; test "$(grep -c ' waits for ' "$g")" -ge 2 && { grep -q '^cycle: ' "$g" || grep -q '^no cycle: ' "$g"; }
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
