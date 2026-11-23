---
type: reference
description: Rules every dispatched agent follows: setup, builds and tests, waiting, git, style and the final report.
---

You are not resumed automatically; block in your own turn until everything is done.

Hard rules: never `pkill` or `killall` (kill only a pid you started, with `kill <pid>`); a wait uses `wait-for.sh`, never `pgrep -f`; no Python. A lint in `verksted_tools` fails on `pkill` and `killall` in scripts.

Rules for every agent in this repo. The ticket section follows below. `brief.sh` prints the values N, SLUG, W, BRANCH, BASE, G, EPIC, PR and VERKSTED; use them wherever a rule says `$N`, `$W`, `$BASE`, `$G`, `$VERKSTED`. `G` is `$GATES` (`lib.sh`). Rule numbers refer to `rules.md`. A path `verksted/...` in the rules lives under `$VERKSTED` before your worktree exists and under `$W/verksted` after; run tools by those absolute paths, never from the main checkout. Each tool prints its usage with `-h`.

Setup and scope
- Read `AGENTS.md` and `docs/issue-workflow.md` first. Work only in the worktree `$W`. Never touch the repo root or another worktree (rule 2).
- Create the worktree with the command named in your prompt (`$VERKSTED/_tools/wt-new.sh <issue> <slug>`, absolute path). Never run workshop tools from the main checkout (rule 61). Never rebase or force-push: when the base moves, `git merge origin/$BASE` (rules 3, 33). If merging the base gives a conflict that is not trivial, stop and report the files.
- Every worktree you create lives in `$WT`; remove your scratch worktrees before you hand back (rule 59).
- Open your pull request against `$BASE`, never `main`: `gh pr create --base $BASE ...`. Do not merge; the orchestrator merges (rule 41).
- Many agents edit the same shared sources in parallel (`src/search.rs`, `src/main.rs`, `src/schema.rs`, `tests/gpu_fit.rs`): keep hunks small and local.

Tests and builds (rules 12, 23, 34)
- Run only the tests your change needs, by exact filter: prove red first, then green. Never the full suite and never the large-store or long benchmark runs; the orchestrator runs one gate on the integration branch (rule 24).
- The integration branch has a zero-warning baseline for the CPU and CUDA builds, and `tests/test_attributes.rs` guards lost `#[test]` attributes. Add no warnings. Before the PR, run the fast lint set once on your head merged with the current base tip, in one build-slot job: `verksted/_tools/lints.sh --tree $W` prints the `--test` arguments. A new test file that spawns processes needs a line in the crate's `tests/sanitizer.plan`.
- Builds use the CPU only. Every cargo command goes through `rc-run.sh`, with `CARGO_BUILD_JOBS=2`, and logs to `$G/issue$N-<tag>.log`. Builds use the build slot (`BUILD_SLOTS` of `machine.md`), test and run use the test lock (one at a time). Both wait while the `PAUSE` file exists, run at nice 19, SCHED_IDLE and ionice idle, and pin work to fewer CPUs while the desktop streams video. Do not work around any of that. The desktop has priority (rule 25).
  `verksted/_tools/rc-run.sh build $W $N build1 --cuda -- cargo test --release --features cuda --no-run --test FILE`
  `verksted/_tools/rc-run.sh test $W $N test1 --cuda -- cargo test --release --features cuda --test FILE FILTER`
  Build first, then test: the test step then only links and runs. Before CUDA tests that spawn `rcsr`, the last build is the CUDA build; before CPU tests, the CPU build (rule 54). A CPU-only change drops `--cuda`; `rc-run.sh test` then runs through the build slot, so it does not queue behind GPU tests. Anything built with `--cuda`, any GPU run and any sanitizer run goes through the lock. The tool refuses a lock hold over 600 s: keep each hold under 10 minutes and give anything that can hang an outer `--timeout`.
- Never run cargo outside `rc-run.sh` and `sanitize.sh`, and never two cargo processes of your own at once. Look first: `ps -eo pcpu,args | grep -E 'cargo|rustc'`.
- Never share one `CARGO_TARGET_DIR` between two worktrees (for example a base build and a branch build): cargo reuses stale `rcsr` artifacts because the package hash does not depend on the path. Each worktree keeps its own `<crate>/target/`.
- Every build and test result you report has a log under `$G` (the tools write it). A result with no log counts as not run. Read counts with `verksted/_tools/test-summary.sh <log>...`, never by eye. After merging the base, rerun and log the tests you report (rule 34).
- A test you report as passed has its own `ok` line in the log; an `#[ignore]` test runs with its variable set (rule 70). `test-summary.sh` prints the counts; `verdict: empty` means nothing ran. A flaky test is a defect: fix the cause, never rerun until green (rule 57).
- Queue a build-slot job only when you need its result and do not have it yet: never a second copy, one job per purpose, test files batched into one job. Build-slot exit 75: requeue once, report a second one (rule 53).
- A change to a hashed byte format runs every golden and pinned-hash test, not only its own (rule 56).
- If your change adds a new refusal or check on a shared path (for example a validation that can reject existing inputs), grep `tests/` for every test that exercises that path and run those files too, so the base never breaks for other agents (rule 36).
- CPU-only checks, scratch calculations and quick looks are shell, `awk`, `bc` or a Rust test.

Waiting (rules 26, 27, 52)
- One wait mechanism: one blocking call, or one background shell and its notification. Never start repeated wait loops.
- One tracked wait per job, as long as the job; never short repeated waits; `wait-for.sh` refuses a second wait on one target (rule 52). Never detach a slot call and hand back.
- Never write `pgrep -f` or `until ! pgrep -f` loops: `pgrep -f` matches the loop's own command line and the loop never ends. Never loop on an exact count that may never match.
- Wait on a log line, a done-file or a process with `verksted/_tools/wait-for.sh --log <file>` (or `--file`, `--pid <N>`, `--exe <abs path>`), which polls inside one process under an outer timeout. A wait uses `wait-for.sh`, never `pgrep -f`. Stop your own wait loops before your final report.
- Never use `pkill` or `killall`. Kill only a process you started.

No Python (rules 9, 49)
- No Python, ever: no `python3`, no `.py` file, no Python fixtures or golden values, and do not type the word `python` as a command, not even as a no-op. A hook blocks any Bash command that invokes `python` or `pip` here. The name is never typed in a shell command, also not in a grep pattern or a commit message typed on a command line (rule 49). Experiments, reference computations and helpers are Rust: crate tests, or single-file programs built with `nice -n 19 rustc -O` outside the repo. Scratch calculations use `awk`, `bc` or a Rust test.

GPU safety (rules 31, 37, 51)
- A device out-of-bounds access can crash the NVIDIA driver and freeze the whole machine. Changed CUDA code, squeezed device memory and allocation-failure tests run first under the sanitizer: `verksted/_tools/sanitize.sh $W $N san1 --test FILE -- FILTER`. It builds the test binary in the slot, runs `$SANITIZER` (`machine.md`) with `--tool memcheck --leak-check no --error-exitcode 99` through the lock with an outer timeout, and checks the Xid count. Use small inputs. No plain GPU run of changed CUDA code until the sanitizer run is clean.
- Search-level GPU tests set `dispatch: DispatchMode::Device` so the kernels really run under memcheck.
- Allocate with `rc_checked_malloc` (`src/cuda/alloc_guard.h`) and call CUB through `RC_CUB` (`src/cuda/cub_guard.h`).
- Before a GPU step the tools record the time; after it they count new `NVRM: Xid` lines since then and exit 70 when there is one: stop all GPU work and report. A falling total is not a fault (rule 31).
- CPU load generators start and stop inside the command the lock runs. A script test never signals a process outside its temp tree, and every lock path in a test points into its temp dir (rule 51).
- Leave no long background job (nohup, multi-hour bench loops) after your final report: it holds the shared lock. Long A/B or bench sweeps belong to the final measurement phase; report what you measured and stop.

Git (rules 4, 5, 32, 33, 48, 50)
- `git commit -m` with a conventional subject (`fix(area): ...`, `feat(area): ...`, `test(area): ...`) and the body line `Refs #N`, plus the epic (`Refs #N, #EPIC`) when the ticket section names one. Never `Closes`, `Fixes` or `Resolves`. No AI attribution, no co-author trailers.
- Never force-push. Never rebase after a push. Never squash or rewrite your branch history (`git reset --soft`, rebase, amend after push): two agents reverted base files that way. Keep plain commits and merge the base; the squash happens at merge time.
- Never use `git stash`: the stash is shared across all worktrees. Prove red with a temporary WIP commit instead.
- Scratch files go in your own `$G/issue$N-scratch/`, never in generic names in shared directories.
- Never bypass a git hook (`--no-verify`, `core.hooksPath`). A hook that blocks a red WIP commit: log the red run instead, or ask the orchestrator (rule 48).
- Never `--edit-last` on an issue or PR comment: edit a comment by its id. Every `gh` call names the repo unless the cwd is a checkout. Before `--body-file F`: `test -s F && grep -q '^## ' F` (rule 50).

Style (rules 10, 11, 15, 72)
- Plain technical English, no em-dashes. Code comments only for constraints the code cannot show; no measured numbers in comments; match the surrounding code. Measured numbers go in the PR body.
- Update `--help`, `rcsr schema`, `README.md` and `docs/` when behaviour, flags or output change.
- Keep the default behaviour bit-identical unless the issue says otherwise (rule 72).

Facts and blockers (rules 13, 35, 39)
- Issue and PR comments, PR bodies and commit bodies state only facts that are true now and that you verified (log path, test name, `git` or `gh`). No predictions or plans stated as facts ("waits for", "will", "belongs to the final test"); write "not run", "open" or "not verified". Attribute any claim you did not check.
- If the issue does not cover a choice, or a criterion fails twice for the same reason, stop and report the exact question.
- If you judge a criterion or the task impossible or too hard, do not just give up: stop and report the exact blocker with the evidence (logs, code paths, what you tried). The orchestrator has it double-checked by a separate research agent.
- A finding outside your issue (bug, regression, warning) is reported, not fixed.
- PR body: what changed, prove-red and green output (commands and counts), deviations from the issue design with reasons, `Refs #N`.

Final report (last message): PR number, head sha, each acceptance criterion with its test name or log path, what you did not verify, deviations with reasons, open items, and any finding outside the issue. Paste `git diff --stat <base>...HEAD`: every file you claim appears there (rule 55).
