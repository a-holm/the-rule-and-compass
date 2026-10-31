# 01_wave_plan: file-disjoint wave under the caps

One job: choose the tickets of this wave so that no two touch the same file.

## Inputs

**This run:**
- `gh issue list --label roadmap --limit 100`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: briefs, dispatch, running agents' worktrees.

## Process

1. List candidate tickets and the files each will touch (from the issue's design section).
2. Pick file-disjoint tickets: at most 10 agents, at most 3 tickets using the machine at once. A ticket whose agent waits on the lock does not count. Tickets that share a file wait for the first to merge. A fourth ticket needs a line `exception: <who allowed it, YYYY-MM-DD>` in `wave.md`.
3. Write `wave.md`: one row per ticket `| #N | file,file | sonnet |`. Research tickets and genuinely difficult implementation run `opus` and use `../../02_research/CONTEXT.md`.
4. Parked issues stay untouched unless the user names them.

## Outputs

- `wave.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
w="$V/03_orchestrate/stages/01_wave_plan/output/wave.md"; r=$(grep -c '^| #[0-9]' "$w"); { test "$r" -ge 1 && test "$r" -le 3; } || grep -q '^exception: ' "$w"; test -z "$(grep '^| #[0-9]' "$w" | cut -d'|' -f3 | tr ', ' '\n\n' | grep . | sort | uniq -d)"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
