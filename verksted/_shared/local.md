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
| `machine.md` | hardware, core map, driver and toolkit versions, the profile keys below, measured throughput with the measurement record behind each number | `10_machine/stages/03_measure` |
| `rules-local.md` | rules that hold on this machine or in this campaign: paths, `TOOLS_WT`, core masks, lock dates, the default epic | the orchestrator |
| `lessons/<type>.md` | lessons that hold for this data or machine | `06_data/stages/09_lessons` |
| `measurements/<date>-<slug>.md` | one record per measurement, method in `measurement.md` | the measuring agent |
| `runs/<run-id>/run.md` | one run card per run | `run-record.sh` |
| `runs/_index.md` | one line per run, newest first, generated | `run-record.sh` |
| `datasets/<id>/` | `card.md`, `adapt.md`, `manifest.md` of a prepared dataset | `06_data/stages/08_verify_manifest` |

A run card holds: id, kind, start and end (UTC), the exact command or the path of the script that holds it, binary sha256 and commit, build features, the stores read with their sha256, split id and seed, CPU mask, wall seconds, user CPU seconds, lock-hours, machine load, new Xid lines, key metrics on development rows, paths of the artefacts outside git, deviations, status.

## Machine profile keys

Tracked text names these keys and never their values. `machine.md` holds one line per key, `- KEY: value`; the scripts that read a key (`resources.sh`) fail with exit 2 and name it when it is missing. `TOOLS_WT` lives in `rules-local.md`, the other keys in `machine.md`.

| Key | Meaning |
|---|---|
| `BUILD_VOLUME` | volume that holds the worktrees and their build targets |
| `DISK_MIN_FREE` | minimum free space on `BUILD_VOLUME` before a new worktree or build |
| `BUILD_SLOT` | the build-slot script |
| `BUILD_SLOTS` | concurrent builds the slot allows |
| `TEST_LOCK` | the test-lock script |
| `PIN_CORES_BUILD` | cores that builds and tests may use |
| `CORES_RESERVED` | cores that pilot, campaign and desktop units keep |
| `CAMPAIGN_BUILD_CAP` | building agents allowed while a GPU campaign runs |
| `BUILD_SLOTS_BUSY` | concurrent builds while the desktop is busy |
| `PIN_CORES_BUSY` | cores for builds and tests while the desktop is busy |
| `MEM_MIN_FREE` | MemAvailable below which no build starts |
| `BUILD_JOBS` | `CARGO_BUILD_JOBS` per build |
| `BUILD_SLOT_WAIT_MAX` | seconds the slot waits before exit 75 |
| `SANITIZER` | the compute-sanitizer that attaches on this driver |
| `CUDA_ARCH` | the device architecture of builds |
| `TOOLS_WT` | a worktree on the integration tip that runs verksted tools before an agent has its own (`rules-local.md`) |

Never in the local area: raw data, a value from inside a locked range before the final seal and reads are posted, secrets, tokens, personal data. Artefacts stay where the run wrote them; a card holds their paths and hashes.

Why ignored: these facts go stale with the machine and the campaign, and a checkout on another machine must not inherit them. Why inside the workshop: stages read them by one variable, and a person finds them next to the contracts. The area is not backed up by git: copy it when the machine changes.

A stage names a local file by its name and this file, never by a machine path.
