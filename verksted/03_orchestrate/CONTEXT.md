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
- `../_templates/brief.md`, `../_templates/brief-research.md`, `../_templates/brief-verifier.md`
- `../_tools/brief.sh`
- `../_tools/status.sh`
- `../_tools/watch.sh`

Do NOT load: the ticket pipelines' stages, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. [01_wave_plan](stages/01_wave_plan/CONTEXT.md): pick file-disjoint tickets, at most 10 agents, at most 3 tickets using the machine (lock waiters do not count).
2. [02_brief](stages/02_brief/CONTEXT.md): one brief per ticket, rendered by `brief.sh` from the templates.
3. [03_dispatch](stages/03_dispatch/CONTEXT.md): dispatch with the model set explicitly, record agent ids.
4. [04_watchdog](stages/04_watchdog/CONTEXT.md): one background watcher, alert after 45 minutes of silence.
5. [05_receive_and_verify](stages/05_receive_and_verify/CONTEXT.md): check every hand-back against PR state, git and logs; a verifier (`brief.sh verifier`) verifies the evidence.
6. [06_merge_cleanup](stages/06_merge_cleanup/CONTEXT.md): after each verifier run, update the integration worktree and report; the merge itself is `../08_close/CONTEXT.md`.

## Outputs

- `stages/06_merge_cleanup/output/report.md`: verdict, tickets merged, tickets open, deviations
- `stages/06_merge_cleanup/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/03_orchestrate/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
