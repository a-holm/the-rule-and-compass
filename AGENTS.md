# AGENTS.md

Instructions for automated agents (and humans) working in this repository.

## What this is

`the-rule-and-compass` is a symbolic-regression engine written in Rust: it
searches for interpretable formulas `y = f(x)` that fit data, ranked on an
error/complexity Pareto front. The crate is in [`my-version/`](my-version/),
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

Recurring agent workflows are numbered pipelines with a checked contract per stage: start at [`verksted/AGENTS.md`](verksted/AGENTS.md).

| Task | Go to |
|---|---|
| implement an issue | [`verksted/01_issue/CONTEXT.md`](verksted/01_issue/CONTEXT.md) |
| research a hard issue | [`verksted/02_research/CONTEXT.md`](verksted/02_research/CONTEXT.md) |
| run agents in parallel | [`verksted/03_orchestrate/CONTEXT.md`](verksted/03_orchestrate/CONTEXT.md) |
| gate and land the integration branch | [`verksted/04_gate/CONTEXT.md`](verksted/04_gate/CONTEXT.md) |
| diagnose a hang or a CPU/GPU delta | [`verksted/05_diagnose/CONTEXT.md`](verksted/05_diagnose/CONTEXT.md) |
| prepare data | [`verksted/06_data/CONTEXT.md`](verksted/06_data/CONTEXT.md) |
| run a long search | [`verksted/07_longrun/CONTEXT.md`](verksted/07_longrun/CONTEXT.md) |
| close an issue and clean up | [`verksted/08_close/CONTEXT.md`](verksted/08_close/CONTEXT.md) |
| report status or hand off | [`verksted/09_handoff/CONTEXT.md`](verksted/09_handoff/CONTEXT.md) |
| protect the shared desktop | [`verksted/10_machine/CONTEXT.md`](verksted/10_machine/CONTEXT.md) |

## Non-negotiables

* Every non-trivial change starts from a GitHub issue. No issue, no work.
* Work happens **only** inside your own `worktrees/issue-<n>-<slug>/` checkout on
  branch `issue/<n>-<slug>`. Never commit directly to `main`.
* `main` must always build and pass `cargo test --release` in `my-version/`.
* Update `--help`, the `schema` output, `README.md` and the relevant file in
  `docs/` whenever behaviour, flags or output change.
* Keep build output, datasets and local scratch out of git. `.gitignore` and the
  local exclude file handle the usual directories — do not add them to a commit.
* Measure performance claims and record the before/after numbers in the issue.
* Rust only. No Python files, no `python3` calls in tests, scripts, docs or workflows. Experiments, reference computations, fixtures and measurement tools are written in Rust; Python's runtime and library behaviour do not show what the Rust code does. The guard is `my-version/tests/no_python.rs`.

## Style

* Rust, pure `std` for the core crate; the CUDA backend is opt-in behind
  `--features cuda`.
* Commit messages follow `feat(area): ...`, `fix(area): ...`, `perf(area): ...`,
  `docs: ...`; reference the issue with `Refs #<n>`. **Never use `Closes #<n>`
  in a commit or PR unless you are closing the issue at that exact moment.**
* Prefer a measurement or a test over an assertion.

## Useful commands

```sh
# build and test the engine
cd my-version && cargo build --release && cargo test --release

# the CLI
cd my-version && ./target/release/rcsr --help
cd my-version && ./target/release/rcsr schema

# start an issue (see docs/issue-workflow.md for the full flow)
git worktree add -b issue/12-metrics-expansion worktrees/issue-12-metrics-expansion main
```
