# 09_handoff: from live state to a safe restart

One job: report the state of the work from authoritative sources, persist it, and make a restart verifiable.

Evidence: status requests and context limits on 2026-10-30 and 2026-10-31, the `/handoff` request of 2026-10-22, `HANDOFF.md` and the memory file for the integration branch. The user rejected "probably" ("hva mener du med sansynligvis, du bør sjekke om du kan", 2026-10-30): check, never guess.

## Inputs

**This run:**
- the trigger: a status question, a context limit or a restart

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/output-format.md`
- `../_templates/status.md`
- `../../AGENTS.md`

Do NOT load: the ticket pipelines' stages, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. `stages/01_state/`: collect the facts, each with its source command.
2. `stages/02_status/`: the status message, outcome first.
3. `stages/03_memory/`: persist the integration state in the memory file.
4. `stages/04_handoff_file/`: write the untracked present-tense `HANDOFF.md`.
5. `stages/05_restart_check/`: after a restart, prove the repository is intact.

## Outputs

- `stages/05_restart_check/output/report.md`: the four restart checks and their results
- `stages/05_restart_check/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/09_handoff/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
