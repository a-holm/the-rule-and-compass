# AGENTS.md

Instructions for automated agents (and humans) working in this repository.

## What this is

`the-rule-and-compass` is a symbolic-regression engine written in Rust: it
searches for interpretable formulas `y = f(x)` that fit data, ranked on an
error/complexity Pareto front. The crate is in `<crate>` (print it with
`verksted/_tools/crate-dir.sh "$W"`),
the specification is [`SPEC.md`](SPEC.md), and the overview is
[`README.md`](README.md).

## Read this first

**Before creating an issue or starting any non-trivial change, read
[`docs/issue-workflow.md`](docs/issue-workflow.md) and follow it exactly.** It
defines:

* how to write and label a detailed issue with **acceptance criteria**;
* the rule **one issue = one branch = one worktree = one pull request**;
* branch naming `issue/<number>-<slug>` and the git-ignored `worktrees/`
  directory that holds each checkout;
* how to open, review and squash-merge a PR, and how to clean up;
* how to run several agents in parallel without them colliding over the same
  files.

For time-series or panel data, read [`docs/time-series.md`](docs/time-series.md)
before choosing a split: it is the checklist for purge/embargo, overlapping
labels, look-ahead preparation, lockboxes, per-group reporting, multiple
testing and transaction costs.

Roadmap work is tracked as GitHub issues labelled `roadmap`:

```sh
gh issue list --label roadmap --limit 100
```

## Hung runs

Run long jobs under `scripts/stall-watch.sh -- <command>`: on a stall it writes all thread backtraces to `<log>.stacks.txt` and never kills anything. Dump a live process with `scripts/stacks.sh <pid>` (the process needs `RCSR_ATTACHABLE=1`). Details: the debugging section of [`docs/issue-workflow.md`](docs/issue-workflow.md).

## Workshop

Recurring agent workflows are numbered pipelines with a checked contract per stage. This file is the single entry point: pick the route, read that `CONTEXT.md`, and stop at the step's `## Check`. Each pipeline folder holds `CONTEXT.md` (Inputs / Process / Outputs / Check) and one `stages/<NN_name>/CONTEXT.md` per stage.

### Where things live

| Folder | What it holds |
|---|---|
| `verksted/01_issue/` | worktree -> run card -> prove red -> implement -> docs -> pull request |
| `verksted/02_research/` | question -> method -> premise -> decision -> issue comment |
| `verksted/03_orchestrate/` | wave plan -> brief -> dispatch -> watchdog -> verify -> merge |
| `verksted/04_gate/` | targeted tests -> integration merge -> one full gate -> tested vs merged -> main |
| `verksted/05_diagnose/` | prove stuck -> stack -> wait graph -> repro -> fix -> medium run -> classify |
| `verksted/06_data/` | intake -> acquire -> profile -> type stage -> panel -> enrich -> audit and lockbox -> manifest -> lessons |
| `verksted/07_longrun/` | preconditions -> launch -> monitor -> resume -> audit -> report |
| `verksted/08_close/` | verifier -> close -> clean up |
| `verksted/09_handoff/` | state -> status -> memory -> HANDOFF.md -> restart check |
| `verksted/10_machine/` | lock -> pause -> measure -> priority -> resume |
| `verksted/_shared/` | factory: commands, rules, agent brief, models, local area, output format, failure modes |
| `verksted/_templates/` | blank starters: a new run is a copy, not a blank page |
| `verksted/_tools/` | every script with its usage and what it writes: [`verksted/_index.md`](verksted/_index.md) |
| `verksted/setup/` | one-time factory configuration |

### Route by what just happened

| If | Go to | Then stop at |
|---|---|---|
| an issue is to be implemented | [`verksted/01_issue/CONTEXT.md`](verksted/01_issue/CONTEXT.md) | the step's `## Check` |
| an issue is hard or unclear | [`verksted/02_research/CONTEXT.md`](verksted/02_research/CONTEXT.md) | the step's `## Check` |
| several tickets run in parallel | [`verksted/03_orchestrate/CONTEXT.md`](verksted/03_orchestrate/CONTEXT.md) | the step's `## Check` |
| tickets are ready for `main` | [`verksted/04_gate/CONTEXT.md`](verksted/04_gate/CONTEXT.md) | the step's `## Check` |
| a GPU test hangs or disagrees with the CPU | [`verksted/05_diagnose/CONTEXT.md`](verksted/05_diagnose/CONTEXT.md) | the step's `## Check` |
| new market data or features, or a new or unknown dataset | [`verksted/06_data/CONTEXT.md`](verksted/06_data/CONTEXT.md) | the step's `## Check` |
| a search runs for hours | [`verksted/07_longrun/CONTEXT.md`](verksted/07_longrun/CONTEXT.md) | the step's `## Check` |
| a PR is merged and the issue may close | [`verksted/08_close/CONTEXT.md`](verksted/08_close/CONTEXT.md) | the step's `## Check` |
| status is asked, or the context ends | [`verksted/09_handoff/CONTEXT.md`](verksted/09_handoff/CONTEXT.md) | the step's `## Check` |
| the machine lags or the user needs it | [`verksted/10_machine/CONTEXT.md`](verksted/10_machine/CONTEXT.md) | the step's `## Check` |
| the previous step is `pass` | the next numbered step | the step's `## Check` |
| which model fits a task | [`verksted/_shared/models.md`](verksted/_shared/models.md) | the tier table |
| the last run, the decisions or the machine profile | [`verksted/_shared/local.md`](verksted/_shared/local.md) | the local area it names |
| status of a run is asked | read `verksted/*/stages/*/output/` and `check.md` | report what exists |
| a link or input path looks wrong | `verksted/_tools/links.sh` | exit 0 |
| which script does what | [`verksted/_index.md`](verksted/_index.md) | the usage line |
| one status screen | `verksted/_tools/status.sh` | the report |
| how the system works | [`docs/how-it-works.md`](docs/how-it-works.md) | the section you need |
| dispatch an agent | `verksted/_tools/brief.sh`, then [`verksted/03_orchestrate/CONTEXT.md`](verksted/03_orchestrate/CONTEXT.md) | the printed prompt |

### The one rule

**Nothing moves to the next step until the previous step's `## Check` is `pass`.** A check that is not a person writes `check.md` beside the output. Its first line is `pass` or `fail: <reason>`. A `fail` stops the run there.

Load only what the step needs: its own `CONTEXT.md`, its `verksted/_shared/` references and this run's inputs. Not the whole workshop.

## Non-negotiables

* Every non-trivial change starts from a GitHub issue. No issue, no work.
* Work happens **only** inside your own `worktrees/issue-<n>-<slug>/` checkout on
  branch `issue/<n>-<slug>`. Never commit directly to `main`.
* `main` must always build and pass `cargo test --release` in `<crate>/`.
* Update `--help`, the `schema` output, `README.md` and the relevant file in
  `docs/` whenever behaviour, flags or output change.
* Keep build output, datasets and local scratch out of git. `.gitignore` and the
  local exclude file handle the usual directories — do not add them to a commit.
* Measure performance claims and record the before/after numbers in the issue.
* Rust only. No Python files, no `python3` calls in tests, scripts, docs or workflows. Experiments, reference computations, fixtures and measurement tools are written in Rust; Python's runtime and library behaviour do not show what the Rust code does. The guard is `<crate>/tests/no_python.rs`.

## Style

* Rust, pure `std` for the core crate; the CUDA backend is opt-in behind
  `--features cuda`.
* Commit messages follow `feat(area): ...`, `fix(area): ...`, `perf(area): ...`,
  `docs: ...`; reference the issue with `Refs #<n>`. **Never use `Closes #<n>`
  in a commit or PR unless you are closing the issue at that exact moment.**
* Prefer a measurement or a test over an assertion.

## Useful commands

```sh
# build and test the engine (<crate> is the output of crate-dir.sh)
cd <crate> && cargo build --release && cargo test --release

# the CLI
cd <crate> && ./target/release/rcsr --help
cd <crate> && ./target/release/rcsr schema

# start an issue: VERKSTED is the absolute path brief.sh prints; never run tools from the main checkout (docs/issue-workflow.md)
"$VERKSTED/_tools/wt-new.sh" 12 metrics-expansion
```
