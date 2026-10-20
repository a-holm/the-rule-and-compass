# Issue workflow

How work is planned and executed in this repository. **Any agent or human
starting an issue must follow this document.** It exists so that work is
traceable, reviewable and parallel-safe.

- [The rules in one list](#the-rules-in-one-list)
- [1. Finding and creating issues](#1-finding-and-creating-issues)
- [2. Starting an issue in its own worktree](#2-starting-an-issue-in-its-own-worktree)
- [3. Doing the work](#3-doing-the-work)
- [4. Opening the pull request](#4-opening-the-pull-request)
- [5. Review and merge](#5-review-and-merge)
- [6. Working in parallel](#6-working-in-parallel)
- [7. House rules](#7-house-rules)

---

## The rules in one list

1. **Every non-trivial change has a GitHub issue first.** If it does not exist,
   create it.
2. **One issue = one branch = one worktree = one pull request.**
3. The branch is named `issue/<number>-<short-slug>` and contains the issue
   number.
4. The worktree lives under the git-ignored `worktrees/` directory, named
   `worktrees/issue-<number>-<short-slug>/`.
5. Issues are **detailed**: context, what is missing, a proposal, and an explicit
   **acceptance criteria** checklist.
6. A pull request is only opened when every acceptance box is true and the tests
   pass.
7. The PR body says `Closes #<number>`.
8. Merge with **squash**, then remove the worktree and delete the branch.
9. Never commit build output, datasets or local scratch. Keep `worktrees/`,
   `target/` and local-only directories out of git (they already are).

---

## 1. Finding and creating issues

List the planned work:

```sh
gh issue list --label roadmap --limit 100
gh issue list --label roadmap --search "sort:created-asc"   # rough priority order
gh issue view <number>
```

Labels:

| label | meaning |
|---|---|
| `roadmap` | planned work from the engine roadmap |
| `area:data` | loading, splitting, preparation |
| `area:expr` | expression tree, operators, evaluator |
| `area:metrics` | error metrics and fitness |
| `area:search` | evolutionary search and problem definition |
| `area:eval` | validation, overfitting, model selection |
| `area:perf` | performance and hardware |
| `area:gpu` | CUDA backend |
| `area:docs` | documentation and reporting |

### Creating a new issue

Use the template (`.github/ISSUE_TEMPLATE/feature.md`) or copy this structure:

```markdown
## Context
Why this matters, in two or three sentences.

## What is missing today
The concrete gap in the current code (name files/functions).

## Proposal
The intended design, with alternatives if there is a real choice.

## Acceptance criteria
- [ ] A testable, concrete outcome.
- [ ] Another one.
- [ ] `cargo test --release` green; `--help`/`schema`/docs updated.

## Notes
Dependencies on other issues, links, measurements.
```

Rules for a good issue:

* Acceptance criteria are **checkable** ("a test shows X", "the JSON contains
  Y"), never "improve Z".
* Name the files and functions the change touches.
* State dependencies (`Depends on #12`).
* If the work is an experiment, say what will be measured.

```sh
gh issue create --title "feat(area): short imperative title" \
  --label "enhancement,roadmap,area:data" --body-file /tmp/body.md
```

---

## 2. Starting an issue in its own worktree

One issue gets its own checkout so that concurrent work never stomps on another
branch.

```sh
cd /path/to/the-rule-and-compass

N=12
SLUG=metrics-expansion          # short, kebab-case, from the issue title

# make sure the base is current
git fetch origin
git switch main && git pull --ff-only

# create branch + worktree in the ignored folder
git worktree add -b "issue/$N-$SLUG" "worktrees/issue-$N-$SLUG" main

# work there from now on
cd "worktrees/issue-$N-$SLUG"
```

`worktrees/` is listed in `.gitignore`, so these checkouts never show up in
`git status` of the main tree. If you use the OpenCode worktree tools instead of
`git worktree`, the same rule applies: the directory must live under
`worktrees/` and the branch must be `issue/<number>-<slug>`.

Leave `target/` at its default inside this worktree. Do **not** point several
worktrees at one `CARGO_TARGET_DIR`: Cargo's fingerprints do not distinguish
two checkouts of the same crate and version, so a build can link an artifact
produced from a different source tree (see §6). The shared lock, not a shared
build directory, is what serialises builds:

```sh
/mnt/lager/tmp/opencode/one-full-suite.sh bash -c \
  'cd worktrees/issue-<n>-<slug>/my-version && cargo test --release'
```

---

## 3. Doing the work

Read the issue's acceptance criteria as the definition of done. Then:

```sh
cd my-version
cargo build --release
cargo test  --release
cargo clippy --release -- -D warnings   # if clippy is installed
```

While working:

* Keep the diff scoped to the issue. Unrelated cleanups go in their own issue.
* Add tests for every acceptance criterion that is testable.
* Update `--help`, the `schema` output, `README.md` and `docs/` when behaviour,
  flags or output change.
* Keep the tree green at every commit; commit in logical steps with messages in
  the existing style (`feat(area): ...`, `fix(area): ...`, `docs: ...`,
  `perf(area): ...`).
* Measure performance claims. If the issue asks for a speed-up, record the
  before/after numbers in the issue (not just the commit message).

```sh
git add -A
git commit -m "feat(metrics): add median, max, rank-correlation and AIC metrics

Closes #12"
```

---

## 4. Opening the pull request

```sh
git push -u origin "issue/$N-$SLUG"

gh pr create --fill --base main \
  --title "$(git log -1 --pretty=%s)" \
  --body "$(cat <<EOF
Closes #$N

## What changed
- ...

## Acceptance criteria
- [x] ... (copied from the issue)

## Verification
- \`cargo test --release\`: N passed
- measured numbers (if performance)

## Notes / follow-ups
- ...
EOF
)"
```

The PR body must repeat the acceptance criteria from the issue with the boxes
ticked, so a reviewer can check them without leaving the page.

---

## 5. Review and merge

Before merging:

1. All acceptance boxes ticked.
2. CI (if configured) green; locally `cargo test --release` green.
3. No unrelated changes in the diff.
4. Output/docs updated.

Then squash-merge and clean up:

```sh
gh pr merge --squash --delete-branch
git switch main && git pull --ff-only
git worktree remove "worktrees/issue-$N-$SLUG"
git branch -d "issue/$N-$SLUG" 2>/dev/null || true
```

Mark the issue's remaining follow-up work as new issues rather than leaving the
original open.

---

## 6. Working in parallel

Parallel work is encouraged **only when the issues do not fight over the same
code**. Before fanning out:

* Give each agent its **own** issue, worktree, branch and PR.
* Prefer issues that touch **different files**. Two agents editing the same file
  will collide at merge time even if the logic is independent.
* If two issues must touch the same file or region, **serialise** them (one PR
  merged before the other starts).
* Never let two agents merge at the same time; merges are done one at a time,
  newest `main` first.

A safe pattern:

```sh
# agent A: issue 12 (src/metrics.rs)
# agent B: issue 6  (src/expr.rs, src/fit.rs)
# agent C: issue 23 (src/diagnostics.rs)
```

Each agent works only inside its own `worktrees/issue-<n>-<slug>/` and pushes only
its own `issue/<n>-<slug>` branch.

### One full-suite test at a time

Parallel *editing* is fine; parallel **building and testing** is not. Each
worktree that keeps its own `my-version/target/` bypasses cargo's own target
lock, so two agents can compile and run the whole suite at once. On an 8-thread
machine that pushed the load average to 13 and made every run slower and noisier.

Rules:

* **Never run `cargo build`, `cargo test` or `rcsr-bench` while another one is
  running.** Look first (`ps -eo pcpu,args | grep -E 'cargo|rustc'`), then wait.
* Run full suites through the shared lock, which queues instead of competing:

  ```sh
  /mnt/lager/tmp/opencode/one-full-suite.sh bash -c \
    'cd worktrees/issue-<n>-<slug>/my-version && cargo test --release'
  ```

* **Never point several worktrees at one `CARGO_TARGET_DIR`.** Cargo's
  fingerprints do not distinguish two checkouts of the same crate and version,
  so a build can link an artifact produced from a *different* source tree. This
  was observed twice in one batch: a stale library and a stale test binary,
  producing bogus compile errors and a test that ran pre-fix code. Each worktree
  keeps its **own** `target/`; the lock provides the serialisation, not a shared
  directory.

* Agents may run **narrow** checks while working (`cargo test --test <name>`,
  `cargo build --release` once after a coherent change). They must not each run
  the full suite at the end, and they must not share a build directory with
  another worktree.
* When several issues are being integrated together, the **combined** branch is
  tested **once**, centrally, by the integrator: one `cargo test --release`, one
  `cargo test --release --features cuda` if the GPU code changed, and one
  `rcsr-bench --tiny`. Individual agents report what they did **not** verify.

### Brief for a delegated agent

> You are working on issue #<n> of `a-holm/the-rule-and-compass`. Read
> `docs/issue-workflow.md` and follow it exactly. Create the worktree
> `worktrees/issue-<n>-<slug>` on branch `issue/<n>-<slug>`, implement the
> acceptance criteria in the issue (`gh issue view <n>`), add tests, commit,
> push, and open a PR that says `Closes #<n>` with the acceptance checklist
> ticked. Do not touch files outside the issue's scope. Report back with the PR
> number and a one-line summary.
>
> **Do not start an independent full-suite run, and do not start one while any
> other `cargo`/`rustc` process is running.** Use narrow checks while you work
> (`cargo test --test <name>`). If a full suite is genuinely required, it must go
> through `/mnt/lager/tmp/opencode/one-full-suite.sh`. The combined integration
> branch is tested once, centrally; in your report, list what you did not
> verify.

---

## 7. House rules

* **Stay in scope.** One issue, one concern.
* **Evidence over opinion.** Claims in an issue or PR are backed by a test, a
  measurement or a reference.
* **Keep the main branch runnable.** Everything merged must build and pass tests.
* **No binary or data blobs in git.** `target/`, `worktrees/` and local scratch
  directories stay ignored.
* **Document behaviour changes.** A flag that exists but is not documented does
  not exist.
* Project overview: [`../README.md`](../README.md); specification:
  [`../SPEC.md`](../SPEC.md).
