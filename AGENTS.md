# AGENTS.md

Instructions for automated agents (and humans) working in this repository.

## What this is

`the-rule-and-compass` is a symbolic-regression engine written in Rust: it searches for interpretable formulas `y = f(x)` that fit data, ranked on an error/complexity Pareto front. The crate is in `<crate>` (print it with `verksted/_tools/crate-dir.sh "$W"`), the specification is [`SPEC.md`](SPEC.md), and the overview is [`README.md`](README.md).

## Read this first

**Before creating an issue or starting any non-trivial change, read [`docs/issue-workflow.md`](docs/issue-workflow.md) and follow it exactly.** It defines the issue format with acceptance criteria, the rule **one issue = one branch = one worktree = one pull request**, branch naming `issue/<number>-<slug>` with the git-ignored `worktrees/` directory, the PR review and squash-merge flow, and how to run agents in parallel.

For time-series or panel data, read [`docs/time-series.md`](docs/time-series.md) before choosing a split. Roadmap work is tracked as GitHub issues labelled `roadmap` (`gh issue list --label roadmap --limit 100`).

## Workshop

The workshop (`verksted/`) describes the integration branch. A copy lives on `main`; paths that exist on the integration branch only (engine crate files, some `docs/` and `scripts/` files) are listed in [`verksted/_shared/main-absent.txt`](verksted/_shared/main-absent.txt). Run its tools from a worktree on the integration branch by the absolute `$VERKSTED` path that `brief.sh` prints, never from the main checkout. Why it is built this way: [`verksted/CONTEXT.md`](verksted/CONTEXT.md).

Recurring workflows are numbered pipelines with a checked contract per stage. This file is the single entry point: pick the route, read that `CONTEXT.md`, and stop at the step's `## Check`. Each pipeline holds `CONTEXT.md` (Inputs / Process / Outputs / Check) and one `stages/<NN_name>/CONTEXT.md` per stage.

### Route by what just happened

| If | Go to | Then stop at |
|---|---|---|
| an issue is to be implemented | [`verksted/01_issue/CONTEXT.md`](verksted/01_issue/CONTEXT.md) | the step's `## Check` |
| a CUDA or GPU change is implemented | [`verksted/01_issue/stages/04_implement/CONTEXT.md`](verksted/01_issue/stages/04_implement/CONTEXT.md), sanitizer first | the step's `## Check` |
| an issue is hard or unclear | [`verksted/02_research/CONTEXT.md`](verksted/02_research/CONTEXT.md) | the step's `## Check` |
| several tickets run in parallel | [`verksted/03_orchestrate/CONTEXT.md`](verksted/03_orchestrate/CONTEXT.md) | the step's `## Check` |
| tickets are ready for `main` | [`verksted/04_gate/CONTEXT.md`](verksted/04_gate/CONTEXT.md) | the step's `## Check` |
| a GPU test hangs or disagrees with the CPU | [`verksted/05_diagnose/CONTEXT.md`](verksted/05_diagnose/CONTEXT.md) | the step's `## Check` |
| new market data or features, or a new or unknown dataset | [`verksted/06_data/CONTEXT.md`](verksted/06_data/CONTEXT.md) | the step's `## Check` |
| a search runs for hours | [`verksted/07_longrun/CONTEXT.md`](verksted/07_longrun/CONTEXT.md) | the step's `## Check` |
| a PR is open and waits for verification | [`verksted/08_close/stages/01_verifier/CONTEXT.md`](verksted/08_close/stages/01_verifier/CONTEXT.md) | the step's `## Check` |
| a PR is merged and the issue may close | [`verksted/08_close/CONTEXT.md`](verksted/08_close/CONTEXT.md) | the step's `## Check` |
| status is asked, or the context ends | [`verksted/09_handoff/CONTEXT.md`](verksted/09_handoff/CONTEXT.md) | the step's `## Check` |
| the machine lags or the user needs it | [`verksted/10_machine/CONTEXT.md`](verksted/10_machine/CONTEXT.md) | the step's `## Check` |
| the previous step is `pass` | the next numbered step | the step's `## Check` |
| which model fits a task | [`verksted/_shared/models.md`](verksted/_shared/models.md) | the tier table |
| the last run, the decisions or the machine profile | [`verksted/_shared/local.md`](verksted/_shared/local.md) | the local area it names |
| a link or input path looks wrong | `verksted/_tools/links.sh` | exit 0 |
| which script does what | [`verksted/_index.md`](verksted/_index.md) | the usage line |
| dispatch an agent | `verksted/_tools/brief.sh`, then [`verksted/03_orchestrate/CONTEXT.md`](verksted/03_orchestrate/CONTEXT.md) | the printed prompt |

**Nothing moves to the next step until the previous step's `## Check` is `pass`.** A check that is not a person writes `check.md` beside the output. Its first line is `pass` or `fail: <reason>`. A `fail` stops the run there. Load only what the step needs: its own `CONTEXT.md`, its `verksted/_shared/` references and this run's inputs.

## Non-negotiables

* Every non-trivial change starts from a GitHub issue. No issue, no work.
* Work happens **only** inside your own `worktrees/issue-<n>-<slug>/` checkout on branch `issue/<n>-<slug>`. Never commit directly to `main`.
* `main` must always build and pass `cargo test --release` in `<crate>/`.
* Update `--help`, the `schema` output, `README.md` and the relevant file in `docs/` whenever behaviour, flags or output change.
* Keep build output, datasets and local scratch out of git. `.gitignore` and the local exclude file handle the usual directories; do not add them to a commit.
* Measure performance claims and record the before/after numbers in the issue.
* Rust only. No Python files, no `python3` calls in tests, scripts, docs or workflows. Experiments, reference computations, fixtures and measurement tools are written in Rust. The guard is `<crate>/tests/no_python.rs`.

## Style

* Rust, pure `std` for the core crate; the CUDA backend is opt-in behind `--features cuda`.
* Commit messages follow `feat(area): ...`, `fix(area): ...`, `perf(area): ...`, `docs: ...`; reference the issue with `Refs #<n>`. **Never use `Closes #<n>` in a commit or PR unless you are closing the issue at that exact moment.**
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

# every long build, test or run goes through the one wrapper (usage: -h)
"$VERKSTED/_tools/rc-run.sh" -h
```
