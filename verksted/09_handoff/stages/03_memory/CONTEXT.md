# 03_memory: persist the integration state

One job: update the memory file with branch contents, running tickets and the queue.

## Inputs

**This run:**
- `../02_status/output/status.md`
- `../02_status/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../setup/questionnaire.md`
- `../../CONTEXT.md`

Do NOT load: history, provenance.

## Process

1. If an input is missing or empty, stop and say which.
2. Update the memory file named in the setup answers (`MEM`): what the integration branch holds, which tickets run, what is queued. Present tense.
3. Write `memory.md` in `output/`: the path and the tickets written.

## Outputs

- `memory.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test -n "$(find "$MEM" -mmin -10 2> /dev/null)" && for n in $(grep -o '#[0-9]\+' "$V/09_handoff/stages/02_status/output/status.md" | sort -u); do grep -q "$n" "$MEM" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
