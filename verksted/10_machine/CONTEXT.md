# 10_machine: share the desktop with the agents

One job: keep the user's desktop first, measure what loads the machine before accusing anyone, and touch only processes you started.

Evidence: the lock script (nice 19, ionice idle, autogroup 19, `PAUSE` file); the 2026-10-30 session where a stream had to be protected, a hung test held the lock for 90 minutes, and a terminal in software rendering used 345 % CPU; "kan du gjøre diskord til hovedprioritet", "stopp crawl.sh" only on request.

## Inputs

**This run:**
- the trigger: the user reports lag or asks for priority

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/failure-modes.md`
- `../_shared/output-format.md`

Do NOT load: the ticket pipelines' stages, earlier runs' `output/`.

## Process

1. `stages/01_lock/`: all heavy work runs through the lock at the lowest priority.
2. `stages/02_pause/`: create the `PAUSE` file so new runs wait.
3. `stages/03_measure/`: measure load, GPU and top consumers.
4. `stages/04_priority/`: act only on processes you started; ask before restarting the terminal.
5. `stages/05_resume/`: remove the `PAUSE` file and report.

## Outputs

- `stages/05_resume/output/report.md`: what was measured, what was done, what is left to the user
- `stages/05_resume/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/10_machine/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
