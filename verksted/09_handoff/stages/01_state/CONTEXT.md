# 01_state: facts with their source

One job: collect the state of PRs, branches, worktrees, gates and processes.

## Inputs

**This run:**
- `gh pr list --state all --limit 30`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: opinions, estimates, agent self-reports.

## Process

1. Run the source commands: `gh pr list`, `gh issue list`, `git log --oneline origin/main..<base>`, `git worktree list`, the gate logs in `$G`, `ps -eo pid,etime,args | grep -E 'cargo|rustc|rcsr'`.
2. Write `state.md`: one bullet per fact, each ending in `(source: <command>)`. A fact without a source is not written.
3. Where something cannot be checked, write `unknown: <what>` and do not guess.

## Outputs

- `state.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
s="$V/09_handoff/stages/01_state/output/state.md"; test "$(grep -c '^- ' "$s")" -ge 1 && test -z "$(grep '^- ' "$s" | grep -v '(source: ' | grep -v '^- unknown: ')"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
