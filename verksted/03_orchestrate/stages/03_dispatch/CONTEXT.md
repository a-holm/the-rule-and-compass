# 03_dispatch: dispatch with an explicit model

One job: start one agent per brief and record who runs what.

## Inputs

**This run:**
- `../02_brief/output/brief-<N>.md`
- `../02_brief/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_tools/brief.sh`
- `../../CONTEXT.md`

Do NOT load: the briefs' content, agents' worktrees.

## Process

1. If an input is missing or empty, stop and say which.
2. Dispatch each brief (the prompt printed by `brief.sh`; its model matches the `--model` it was rendered with) with the Agent tool and set `model` explicitly: `sonnet` to implement, rebase, salvage or verify; `opus` only for hard or demanding tasks (research on unclear issues, genuinely difficult implementation), always with tractatus thinking and context7, never web search or web fetch.
3. Record `dispatch.md`: rows `| #N | <agent id> | sonnet |` and the dispatch time.
4. A dead agent's unfinished work is salvaged by a new Sonnet agent that treats it as an unreviewed draft; the orchestrator does not edit it.
5. An agent that judges its task impossible or too hard reports the exact blocker with evidence. Dispatch an Opus agent (tractatus thinking and context7, no web search or web fetch) to double-check before the verdict is accepted (rule 39).
6. An Opus agent that stops on an API error is rerun on Sonnet with the same brief (rule 40).

## Outputs

- `dispatch.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
d="$V/03_orchestrate/stages/03_dispatch/output/dispatch.md"; w="$V/03_orchestrate/stages/01_wave_plan/output/wave.md"; for n in $(grep -o '^| #[0-9]*' "$w" | tr -d '| #'); do grep -qE "^\| #$n \| [^ |]+ \| (sonnet|opus) \|" "$d" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
