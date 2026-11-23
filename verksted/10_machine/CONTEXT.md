# 10_machine: share the desktop with the agents

One job: keep the user's desktop first, measure what loads the machine before accusing anyone, and touch only processes you started.

Evidence: the lock scripts (SCHED_IDLE, nice 19, ionice idle, autogroup 19, streaming-aware core pinning, `PAUSE` file); the 2026-10-30 session where a stream had to be protected, a hung test held the lock for 90 minutes, and a terminal in software rendering used 345 % CPU; "kan du gjøre diskord til hovedprioritet", "stopp crawl.sh" only on request.

## Inputs

**This run:**
- the trigger: the user reports lag or asks for priority

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/failure-modes.md`
- `../_shared/output-format.md`
- `../_tools/status.sh`
- `../_tools/machine-sync.sh`

Do NOT load: the ticket pipelines' stages, earlier runs' `output/`.

## Process

1. [01_lock](stages/01_lock/CONTEXT.md): builds in the build slot, tests and GPU runs behind the test lock, live scripts checked against the canonical copies.
2. [02_pause](stages/02_pause/CONTEXT.md): create the `PAUSE` file so new runs wait.
3. [03_measure](stages/03_measure/CONTEXT.md): measure load, GPU and top consumers.
4. [04_priority](stages/04_priority/CONTEXT.md): the desktop has priority; screen blanking and lock stay off; act only on processes you started; ask before restarting the terminal.
5. [05_resume](stages/05_resume/CONTEXT.md): remove the `PAUSE` file and report.

## Outputs

- `stages/05_resume/output/report.md`: what was measured, what was done, what is left to the user
- `stages/05_resume/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/10_machine/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
