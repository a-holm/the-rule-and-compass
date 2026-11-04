# 05_restart_check: the repository is intact

One job: prove the repository and the machine are in the recorded state.

## Inputs

**This run:**
- `../04_handoff_file/output/handoff.md`
- `../04_handoff_file/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`
- `../../../_tools/status.sh`

Do NOT load: continuing work before all four checks pass.

## Process

1. If an input is missing or empty, stop and say which.
2. Run `verksted/_tools/status.sh --no-fetch`. Four checks read from its lines: the `origin/<base>` tip equals the branch tip, the run worktree shows `dirty=n`, the `processes:` list is `none`, `flags:` shows `PAUSE=n`.
3. Write `report.md`: each check with its command and result.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
s=$("$V/_tools/status.sh" --no-fetch); test "$(printf "%s\n" "$s" | sed -n "s|^origin/$BASE: \([0-9a-f]*\) .*|\1|p")" = "$(git -C "$RUN" rev-parse --short HEAD)" && printf "%s\n" "$s" | grep -q "^$(basename "$RUN"): .* dirty=n" && test "$(printf "%s\n" "$s" | sed -n "/^processes:/,\$p" | tail -n +2)" = "  none" && printf "%s\n" "$s" | grep -q "^flags: PAUSE=n"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
