# 03_dispatch: dispatch with an explicit model

One job: start one agent per brief and record who runs what.

## Inputs

**This run:**
- `../02_brief/output/brief-<N>.md`
- `../02_brief/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_tools/brief.sh`
- `../../../_shared/models.md`
- `../../CONTEXT.md`

Do NOT load: the briefs' content, agents' worktrees.

## Process

1. If an input is missing or empty, stop and say which.
2. Dispatch each brief (the prompt printed by `brief.sh`; its model matches the `--model` it was rendered with) with the Agent tool and set the model or agent type explicitly, by the tiers of `models.md`: `sonnet` to execute (implement, rebase, salvage, verify); `opus` to plan; `subagent_type: fable-52` to plan an extraordinarily hard case. A planning agent uses tractatus thinking, context7 and web search whenever it helps. Caps and cwd: rules 21, 61 and 66 (at most 10 agents and 3 tickets, no git or cargo with the main checkout as cwd, at most `CAMPAIGN_BUILD_CAP` building agents while a GPU campaign runs).
3. Record `dispatch.md`: rows `| #N | <agent id> | sonnet |` (the last column is `sonnet`, `opus` or `fable-52`) and the dispatch time.
4. A dead agent's unfinished work is salvaged by a new Sonnet agent that treats it as an unreviewed draft; the orchestrator does not edit it.
5. An agent that judges its task impossible or too hard reports the exact blocker with evidence. Dispatch an Opus agent (tractatus thinking, context7 and web search) to double-check before the verdict is accepted (rule 39).
6. A planning agent that stops on an API error is rerun on the next tier down with the same brief (`models.md`).

## Outputs

- `dispatch.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
d="$V/03_orchestrate/stages/03_dispatch/output/dispatch.md"; w="$V/03_orchestrate/stages/01_wave_plan/output/wave.md"; test -s "$d" && test -s "$w" && test -n "$(grep -o '^| #[0-9]*' "$w")" && for n in $(grep -o '^| #[0-9]*' "$w" | tr -d '| #'); do grep -qE "^\| #$n \| [^ |]+ \| (sonnet|opus|fable-52) \|" "$d" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
