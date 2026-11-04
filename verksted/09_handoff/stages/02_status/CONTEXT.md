# 02_status: outcome first

One job: write the status message from the state file.

## Inputs

**This run:**
- `../01_state/output/state.md`
- `../01_state/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_templates/status.md`
- `../../CONTEXT.md`

Do NOT load: facts that are not in state.md.

## Process

1. If an input is missing or empty, stop and say which.
2. Copy `_templates/status.md` to `output/status.md`. Line 1 is `Outcome: <what happened>`.
3. One row per ticket: `| #N | PR | state | gate | next |`. Every ticket that appears in `state.md` appears here.
4. Answer the user in their language and extremely concise. The file stays English.

## Outputs

- `status.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
a="$V/09_handoff/stages/02_status/output/status.md"; head -1 "$a" | grep -q '^Outcome: ' && for n in $(grep -o '#[0-9]\+' "$V/09_handoff/stages/01_state/output/state.md" | sort -u); do grep -q "^| $n |" "$a" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
