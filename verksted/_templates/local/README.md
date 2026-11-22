# Local area

Git-ignored state of this machine and campaign. Layout and rules: `verksted/_shared/local.md`. Created by `verksted/_tools/local-init.sh`; the script never overwrites a file.

| Path | Holds | Written by |
|---|---|---|
| `README.md` | this layout, copied | `local-init.sh` |
| `decisions.md` | decisions register of the running campaign | the orchestrator |
| `machine.md` | hardware, core map, versions, measured throughput | `10_machine/stages/03_measure` |
| `rules-local.md` | rules for this machine or campaign: paths, core masks, lock dates | the orchestrator |
| `lessons/<type>.md` | lessons that hold for this data or machine | `06_data/stages/09_lessons` |
| `measurements/<date>-<slug>.md` | one record per measurement | the measuring agent |
| `runs/<run-id>/run.md` | one run card per run | `run-record.sh` |
| `runs/_index.md` | one line per run, newest first, generated | `run-record.sh` |
| `datasets/<id>/` | `card.md`, `adapt.md`, `manifest.md` of a prepared dataset | `06_data/stages/08_verify_manifest` |
