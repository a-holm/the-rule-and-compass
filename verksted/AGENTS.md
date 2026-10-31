# AGENTS.md: workshop

The workshop is the ICM workspace for the recurring agent workflows of this repo. Each pipeline is a numbered sequence. Each folder has a `CONTEXT.md` with Inputs / Process / Outputs / Check.

## Where things live

| Folder | What it holds |
|---|---|
| `01_issue/` | worktree -> run card -> prove red -> implement -> docs -> pull request |
| `02_research/` | question -> method -> premise -> decision -> issue comment |
| `03_orchestrate/` | wave plan -> brief -> dispatch -> watchdog -> verify -> merge |
| `04_gate/` | targeted tests -> integration merge -> one full gate -> tested vs merged -> main |
| `05_diagnose/` | prove stuck -> stack -> wait graph -> repro -> fix -> medium run -> classify |
| `06_data/` | acquire -> import/prep -> panel -> enrich -> audit and lockbox -> manifest |
| `07_longrun/` | preconditions -> launch -> monitor -> resume -> audit -> report |
| `08_close/` | verifier -> close -> clean up |
| `09_handoff/` | state -> status -> memory -> HANDOFF.md -> restart check |
| `10_machine/` | lock -> pause -> measure -> priority -> resume |
| `_shared/` | factory: commands, rules, agent brief, output format, failure modes |
| `_templates/` | blank starters: a new run is a copy, not a blank page |
| `_tools/` | `links.sh`, `check.sh`, `gate.sh`, `watch.sh` |
| `setup/` | one-time factory configuration |

## Route by what just happened

| If | Go to | Then stop at |
|---|---|---|
| an issue is to be implemented | `01_issue/CONTEXT.md` | the step's `## Check` |
| an issue is hard or unclear | `02_research/CONTEXT.md` | the step's `## Check` |
| several tickets run in parallel | `03_orchestrate/CONTEXT.md` | the step's `## Check` |
| tickets are ready for `main` | `04_gate/CONTEXT.md` | the step's `## Check` |
| a GPU test hangs or disagrees with the CPU | `05_diagnose/CONTEXT.md` | the step's `## Check` |
| new market data or features | `06_data/CONTEXT.md` | the step's `## Check` |
| a search runs for hours | `07_longrun/CONTEXT.md` | the step's `## Check` |
| a PR is merged and the issue may close | `08_close/CONTEXT.md` | the step's `## Check` |
| status is asked, or the context ends | `09_handoff/CONTEXT.md` | the step's `## Check` |
| the machine lags or the user needs it | `10_machine/CONTEXT.md` | the step's `## Check` |
| the previous step is `pass` | the next numbered step | the step's `## Check` |
| status of a run is asked | read `*/stages/*/output/` and `check.md` | report what exists |
| a link or input path looks wrong | `verksted/_tools/links.sh` | exit 0 |

## The one rule

**Nothing moves to the next step until the previous step's `## Check` is `pass`.** A check that is not a person writes `check.md` beside the output. Its first line is `pass` or `fail: <reason>`. A `fail` stops the run there.

Load only what the step needs: its own `CONTEXT.md`, its `_shared/` references and this run's inputs. Not the whole workshop.
