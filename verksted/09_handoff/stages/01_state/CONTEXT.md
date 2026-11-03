# 01_state: facts with their source

One job: collect the state of PRs, branches, worktrees, gates and processes.

## Inputs

**This run:**
- `gh pr list --state all --limit 30`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`
- `../../../_tools/status.sh`

Do NOT load: opinions, estimates, agent self-reports.

## Process

1. Run `verksted/_tools/status.sh`: it prints the clock, the base tip, open PRs, worktrees, locks, flags, GPU and processes. Add `gh pr list`, `gh issue list`, `git log --oneline origin/main..<base>` and the gate logs in `$GATES`.
2. Write `state.md`: one bullet per fact, each ending in `(source: status.sh <section>)` or `(source: <command>)`. A fact without a source is not written. The integration branch never goes to `main` as a whole. The one approved exception is `verksted/` alone, copied to `main` as its own issue after epic #666 is done. Any other merge into `main` needs the project lead's written instruction (standing instruction, 2026-11-03).
3. Where something cannot be checked, write `unknown: <what>` and do not guess.

## Outputs

- `state.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
s="$V/09_handoff/stages/01_state/output/state.md"; test "$(grep -c '^- ' "$s")" -ge 1 && test -z "$(grep '^- ' "$s" | grep -v '(source: ' | grep -v '^- unknown: ')"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
