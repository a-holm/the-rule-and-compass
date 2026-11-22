---
type: reference
description: Layout and rules of the git-ignored local area that holds machine, campaign and run state.
---

# The local area

Tracked files of the workshop are general. Everything that holds for one machine, one campaign or one session lives in the local area, which git ignores (`.gitignore` of the workshop: `/_local/`).

Where: `$VLOCAL`, defined in `_tools/lib.sh` as `verksted/_local/` of the main checkout, so every worktree uses the same directory. `RCSR_LOCAL` overrides it. Create it with `verksted/_tools/local-init.sh`; the script copies the starters of `_templates/local/` and never overwrites a file.

| Path under `$VLOCAL` | Holds | Written by |
|---|---|---|
| `README.md` | this layout, copied | `local-init.sh` |
| `decisions.md` | the decisions register of the running campaign: id, date, who decided, the decision, the reason, what it replaces | the orchestrator |
| `machine.md` | hardware, core map, driver and toolkit versions, measured throughput with the measurement record behind each number | `10_machine/stages/03_measure` |
| `rules-local.md` | rules that hold on this machine or in this campaign: paths, core masks, lock dates | the orchestrator |
| `lessons/<type>.md` | lessons that hold for this data or machine | `06_data/stages/09_lessons` |
| `measurements/<date>-<slug>.md` | one record per measurement, method in `measurement.md` | the measuring agent |
| `runs/<run-id>/run.md` | one run card per run | `run-record.sh` |
| `runs/_index.md` | one line per run, newest first, generated | `run-record.sh` |
| `datasets/<id>/` | `card.md`, `adapt.md`, `manifest.md` of a prepared dataset | `06_data/stages/08_verify_manifest` |

A run card holds: id, kind, start and end (UTC), the exact command or the path of the script that holds it, binary sha256 and commit, build features, the stores read with their sha256, split id and seed, CPU mask, wall seconds, user CPU seconds, lock-hours, machine load, new Xid lines, key metrics on development rows, paths of the artefacts outside git, deviations, status.

Never in the local area: raw data, a value from inside a locked range before the final seal and reads are posted, secrets, tokens, personal data. Artefacts stay where the run wrote them; a card holds their paths and hashes.

Why ignored: these facts go stale with the machine and the campaign, and a checkout on another machine must not inherit them. Why inside the workshop: stages read them by one variable, and a person finds them next to the contracts. The area is not backed up by git: copy it when the machine changes.

A stage names a local file by its name and this file, never by a machine path.
