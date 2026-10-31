# 04_watchdog: one watcher, no wait loops

One job: start the silence watcher over the agents' worktrees.

## Inputs

**This run:**
- `../03_dispatch/output/dispatch.md`
- `../03_dispatch/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_tools/watch.sh`
- `../../CONTEXT.md`

Do NOT load: agent transcripts (they buffer and prove nothing about liveness).

## Process

1. If an input is missing or empty, stop and say which.
2. Start `verksted/_tools/watch.sh` once in the background over every agent worktree and write its pid to `output/watch.pid`. It alerts after 45 minutes without a file change unless cargo, flock, nvcc or rcsr runs under that path.
3. Wait for the background notification. Never write a `pgrep -f` loop or a `sleep` loop of your own.
4. An alert is a question to the agent (SendMessage), not a kill.

## Outputs

- `watch.pid` -> `output/`
- `check.md` -> `output/`

## Check

```sh
kill -0 "$(cat "$V/03_orchestrate/stages/04_watchdog/output/watch.pid")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
