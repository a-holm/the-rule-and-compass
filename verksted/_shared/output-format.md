# Output format: check.md, reports and names

## Every step writes to output/

A step's product is a file in its own `output/` folder, not a chat message. `output/` is git-ignored (`verksted/**/output/`); the workshop itself is tracked.

- The run root (`RUN`) is the checkout that holds the run's `output/` files: the ticket worktree for ticket pipelines, the orchestrator's integration worktree for orchestration, gate, close, handoff and machine pipelines.
- A new run does not inherit the previous run's `output/`. Empty it first.
- A missing input is not an empty product: the step names what is missing and stops.
- Status is what exists on disk: a stage is complete when its `output/` holds the artifact and a `check.md` that says `pass`.

## check.md

Written beside the output by `_tools/check.sh`, or by hand for a check that is a person.

```
pass

- command: `<command>`
- exit: 0
```

or

```
fail: exit 1: <last output line of the command>

- command: `<command>`
- exit: 1
```

Line 1 is exactly `pass` or `fail: <reason>`. Later lines are evidence. A `fail` stops the run at that stage. A `fail` is never edited into a `pass`: fix the output, rerun the check. `check.md` is never committed.

## Report format

The last stage of every pipeline writes `report.md`.

```
# <pipeline>: <issue or run>

## Verdict
pass | fail: <reason>

## Evidence
<commands, exit codes and numbers, not claims>

## Deviations
<everything skipped, with the reason; "none" only if nothing was skipped>

## Next step
<one concrete action, or "none">
```

- The verdict is one line. "Probably green" is not a verdict.
- Evidence is a command with its result, or a log path with its count.
- Silence about a skipped step is an error.

## Other artefacts

| Artefact | Starter | Rule |
|---|---|---|
| issue body | `../_templates/issue-body.md` | metadata table, measurable criteria, dependencies |
| run card | `../_templates/run.md` | one proof line per acceptance criterion |
| pull request body | `../_templates/pr.md` | `Refs #N`, never `Closes` |
| research question, decision | `../_templates/research-question.md`, `../_templates/research-decision.md` | premises, then a binding spec |
| agent brief | `brief.md` + `../_templates/brief.md` | starts with the no-resume line |
| verification comment | `../_templates/verification-comment.md` | one row per criterion |
| status message | `../_templates/status.md` | outcome first |

## Names

- Stage folders: `NN_snake_case`. Pipeline folders: `NN_noun`.
- Worktree `worktrees/issue-<n>-<slug>`, branch `issue/<n>-<slug>`, slug kebab-case from the issue title.
- Logs: `/mnt/lager/tmp/eureqa-gates/issue<n>-<what>.log`, gate logs `<tag>-{cpu,cuda,bench}.log`.
- Dates `YYYY-MM-DD`, never "today".
