# 03_orchestrate: from open tickets to merged branches

One job: run several tickets in parallel without collisions, and verify every hand-back against GitHub and git.

Evidence: 39 agent briefs in one orchestrator session, 394 subagent dispatches in the opencode root session, 4 tickets active against a cap of 3 once, 57 duplicate wait loops from one agent, stray `pkill` calls. The orchestrator never works inside an agent's worktree.

## Inputs

**This run:**
- the list of open tickets (`gh issue list --label roadmap --limit 100`) and the integration branch name `BASE`

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/brief.md`
- `../_shared/output-format.md`
- `../_templates/brief.md`
- `../_tools/watch.sh`

Do NOT load: the ticket pipelines' stages, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. `stages/01_wave_plan/`: pick file-disjoint tickets, at most 10 agents, at most 3 tickets using the machine (lock waiters do not count).
2. `stages/02_brief/`: one brief per ticket from the template.
3. `stages/03_dispatch/`: dispatch with the model set explicitly, record agent ids.
4. `stages/04_watchdog/`: one background watcher, alert after 45 minutes of silence.
5. `stages/05_receive_and_verify/`: check every hand-back against PR state, git and logs.
6. `stages/06_merge_cleanup/`: merge into `BASE` one at a time, clean up path by path.

## Outputs

- `stages/06_merge_cleanup/output/report.md`: verdict, tickets merged, tickets open, deviations
- `stages/06_merge_cleanup/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/03_orchestrate/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
