# 04_priority: act only on my own processes

One job: act on processes you started, ask about everything else.

## Inputs

**This run:**
- `../03_measure/output/measure.md`
- `../03_measure/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/rules.md`
- `../../CONTEXT.md`

Do NOT load: foreign processes, the terminal, the user's crawl.

## Process

1. If an input is missing or empty, stop and say which.
2. For each consumer decide: mine, the user's or the system's. Only mine may be stopped, and only with `kill <pid>` of a pid you started. Never `pkill` or `killall`.
3. Restarting the terminal or multiplexer ends this session: ask first, and fix permanently so the user restarts in their own time.
4. Write `actions.md`: rows `| pid | owner | started by me | action |`; the action is `none` when not mine.

## Outputs

- `actions.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
a="$V/10_machine/stages/04_priority/output/actions.md"; test "$(grep -c '^| [0-9]' "$a")" -ge 1 && ! awk -F'|' '/^\| [0-9]/{gsub(/ /,"",$4);gsub(/ /,"",$5); if($4!="yes" && $5!="none") bad=1} END{exit !bad}' "$a"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
