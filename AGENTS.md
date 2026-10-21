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
