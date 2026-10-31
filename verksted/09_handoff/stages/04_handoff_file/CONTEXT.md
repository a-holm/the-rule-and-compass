# 04_handoff_file: present tense, untracked

One job: write HANDOFF.md for the next session.

## Inputs

**This run:**
- `../03_memory/output/memory.md`
- `../03_memory/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: history of how the state came about.

## Process

1. If an input is missing or empty, stop and say which.
2. Write `$RUN/HANDOFF.md`: what the project does, what is open, what to research next, the commands to continue. No history, no provenance.
3. Do not commit the file. It is temporary.

## Outputs

- `handoff.md` -> `output/` (a copy of the sections written)
- `check.md` -> `output/`

## Check

```sh
test "$(git -C "$RUN" status --porcelain -- HANDOFF.md)" = '?? HANDOFF.md' && ! grep -qiE 'as of [0-9]|previously|session|claude|generated' "$RUN/HANDOFF.md"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
