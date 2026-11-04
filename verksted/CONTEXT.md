# workshop: ten measured workflows

The workshop is an ICM workspace (Interpretable Context Methodology). The folders are the orchestration, `CONTEXT.md` is the contract, and status is what exists on disk. Repo rules: `../AGENTS.md` and `../docs/issue-workflow.md`.

## Why these pipelines

Each pipeline is a job that repeated in the agent logs: one Claude Code orchestrator session (2026-10-23 to 2026-10-31, 39 briefs, 36 subagents, 2779 subagent shell commands) and the opencode sessions (303 sessions in this repo, 2026-10-18 to 2026-10-31).

| Pipeline | Evidence | Why its own pipeline |
|---|---|---|
| `01_issue/` | 39 briefs; PRs up to #369; 14 (Claude Code) and 29 (opencode) `gh pr merge` calls | one ticket, one branch, one PR; red before green |
| `02_research/` | #254, #257, #291, #292, #293, #352, #364 | Opus, no code, ends in a comment, not a PR |
| `03_orchestrate/` | 394 dispatches; 4 tickets against a cap of 3; 57 duplicate wait loops | the orchestrator never works in an agent's worktree |
| `04_gate/` | 20 full gates for a handful of PRs; user correction 2026-10-30 | one gate for many tickets |
| `05_diagnose/` | #303, #351, #352, #353, #355, #357, #359, #364 | a different loop: measure, stack, graph, repro |
| `06_data/` | crypto spot work 2026-10-19 to 2026-10-22; 41 retired scripts | data order matters; leakage checks |
| `07_longrun/` | the 10-hour run; eight `rcsr` processes where one was meant | preconditions, resume, audit |
| `08_close/` | `Closes` auto-closing; the #257 verifier | closing is a separate act by another agent |
| `09_handoff/` | `/handoff` 2026-10-22; status requests 2026-10-30 | facts with a source, then a restart check |
| `10_machine/` | lock, `PAUSE`, desktop priority 2026-10-30 | etiquette on a shared machine |

## Deliberately outside

- Rust-only experiments: no pipeline. They are one rule (`_shared/rules.md` rule 9), one command block (`_shared/commands.md`, "Rust experiments") and one test (`my-version/tests/no_python.rs`). Stage `01_issue/stages/03_prove_red/` and `01_issue/stages/04_implement/` apply it.
- Product-specific one-off research projects: they do not repeat.
- Issue creation: a template (`_templates/issue-body.md`) and the checks in `01_issue/stages/02_issue_body/`, not a pipeline.

## The flow in one line

```
01_issue: worktree -> issue_body -> prove_red -> implement -> docs_sync -> pr
02_research: question -> method -> premise -> decision -> issue_comment
03_orchestrate: wave_plan -> brief -> dispatch -> watchdog -> receive_and_verify -> merge_cleanup
04_gate: targeted_tests -> integration_merge -> full_gate -> tested_vs_merged -> merge_main
05_diagnose: prove_stuck -> stack -> wait_graph -> repro_red -> fix_bounded -> medium_run -> classify_deltas
06_data: acquire -> import_prep -> panel_pit -> enrich -> audit_lockbox -> verify_manifest
07_longrun: preconditions -> launch -> monitor -> resume -> audit -> report
08_close: verifier -> close -> cleanup
09_handoff: state -> status -> memory -> handoff_file -> restart_check
10_machine: lock -> pause -> measure -> priority -> resume
```

Factory (stable, every run): `_shared/commands.md`, `_shared/rules.md`, `_shared/brief.md`, `_shared/output-format.md`, `_shared/failure-modes.md`, `_templates/`, `_tools/`.
Product (new each run): each stage's `output/` (git-ignored).

A run is set up once from `setup/questionnaire.md`. Its answers are already written to `_shared/` and are not asked again.

## Check the structure itself

```sh
verksted/_tools/links.sh [--strict]
```

It builds the crate `_tools/verksted-check/` in the build slot and runs `links` and `walk` on `verksted/`. `links` checks that every relative link and every backticked input path resolves. `walk` checks that every pipeline and stage holds the four contract headings in order, that `AGENTS.md` routes every pipeline, that no `.py` file exists, that `CLAUDE.md` is exactly `@AGENTS.md`, that every script header follows the convention below, and that every `_tools/<path>` mention in a markdown file exists. `--strict` adds three checks: `_index.md` matches the tree, every tool is mentioned in a file other than `_index.md`, and the repo `AGENTS.md` links `verksted/AGENTS.md`. The non-strict rules run in `cargo test --release` in `my-version/` through `tests/verksted_links.rs`, the strict rules through `tests/verksted_walk.rs` (run `--test verksted_walk` after adding or renaming a tool); `tests/verksted_tools.rs` runs `bash -n` on every script and every `_tools/tests/*.sh`. Exit 0 means the tree is clean.

## Scripts

Every top-level `_tools/*.sh` follows one convention:

- Line 1 is `#!/usr/bin/env bash`. Line 2 starts with `# Usage:`, or with `# Sourced` for a file that is only sourced. A `# Writes:` line sits in the first 15 lines.
- It starts with `set -uo pipefail` and `source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"`. `lib.sh` holds the paths, the `RCSR_*` overrides and the helpers; no script repeats a machine path.
- `-h` prints the header (`usage`). A bad argument prints it to stderr and exits 2.
- Exit codes: 0 ok, 1 failed or refused, 2 usage, 70 a GPU Xid error appeared, 75 the lock gave up, 124 timeout.
- A script that runs a job prints `RC_DONE rc=<n> at=<time>` as its last line (`rc_done`).
- A script never creates `$GATES`; it calls `need_gates` before it writes a log.
- Each script has a hermetic test in `_tools/tests/test-<name>.sh`: no network, no GPU, no real lock, no real `gh`, temp dirs only, no real timer and no wall-clock check. A wait ends on an event (`hold` in `_tools/tests/lib/fakes.sh`) or on a counted poll of the fake clock; `tools_tests_read_no_clock` enforces it. `tests/verksted_tools.rs` runs every `_tools/tests/test-*.sh`; a script fails on a non-zero exit, or when its whole process tree stays blocked with nothing running for 30 snapshots.

## Index

`_index.md` lists the pipelines, stages, tools and templates. It is generated: never edit it by hand. After a change to a stage, a tool or a template, run `verksted-check index` (`_tools/links.sh` builds the binary). `links.sh --strict` fails when the file is stale.
