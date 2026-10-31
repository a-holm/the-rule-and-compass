# Rules: hard requirements for every pipeline

The rules most often broken, each with its source. Sources: `AGENTS.md` and `docs/issue-workflow.md` of this repo, the user's global rules, and dated user instructions ("user, YYYY-MM-DD"). Failures behind them: `failure-modes.md`.

## Work surface

1. One issue, one worktree, one branch, one pull request. Branch `issue/<n>-<slug>`, worktree `worktrees/issue-<n>-<slug>/`. (source: `AGENTS.md` Non-negotiables; `docs/issue-workflow.md` rules 2 to 4)
2. Work only in your own worktree. Never touch the repo root or another agent's worktree. The orchestrator delegates fixes back to the agent or dispatches a new one. (source: `AGENTS.md`; user, 2026-10-21; global orchestration rules)
3. Never commit to `main`. Never force-push. Never rebase after a push: merge the base branch into yours. (source: `AGENTS.md`; the 2026-10-30 refusal of a `--force-with-lease` brief)
4. `Refs #<n>`, never `Closes`, `Fixes` or `Resolves`. (source: `AGENTS.md` Style; `docs/issue-workflow.md` rule 7; user, 2026-10-21)
5. Commits follow the repo style `feat(area): ...`, `fix(area): ...`, `perf(area): ...`, `docs: ...` with a `Refs #<n>` body line. This repo style replaces the global `#<n>: ` subject prefix. Squash subjects are hand-written: `type(area): subject (#<pr>)`. No AI attribution, no co-author trailers. (source: `AGENTS.md` Style; global commit rules)
6. Each worktree keeps its own `my-version/target/`. Never share a `CARGO_TARGET_DIR`. (source: `docs/issue-workflow.md` sections 2 and 6; stale artefacts seen twice, 2026-10-18)
7. Clean up path by path, only the finished ticket's worktree and branches. No `rm -rf`, no bulk clean-up while agents live. (source: global orchestration rules; denied removals, 2026-10-30)
8. Keep scratch, datasets and build output out of git. (source: `AGENTS.md` Non-negotiables)

## Language and tools

9. No Python. No `.py` file, no `python3` call, no python shebang. Experiments, reference values, fixtures and helpers are Rust: crate tests, or single-file `rustc -O` programs outside the repo. Quick checks are shell. Scripted edits use the Edit tool, `perl` or `sed`. (source: `docs/issue-workflow.md` rule 10; `my-version/tests/no_python.rs`; user, 2026-10-31)
10. Plain technical English, short sentences, no em-dashes. Docs describe the present only: no history, no provenance, no AI references. (source: global writing rules)
11. Code comments state a constraint the code cannot show. No measured numbers in comments. (source: global commit and ticket rules)

## Quality

12. Prove red first, then green: the new test fails on the unfixed code. Measure before building; label every claim measured, derived or inferred. (source: `AGENTS.md` "Prefer a measurement or a test over an assertion"; user, 2026-10-30 "check, never guess")
13. Verify against PR state, `git log` and log files, never an agent's self-report. An inferred claim is not evidence. (source: global orchestration rules)
14. Reported values are exact CPU f64. Never weaken an exactness guard. Never claim GPU work that did not execute. (source: user, 2026-10-18 and 2026-10-19 "exactness is a goal")
15. Update `--help`, `rcsr schema`, `README.md` and `docs/` when behaviour, flags or output change. (source: `AGENTS.md` Non-negotiables)
16. Close an issue only after the six closure conditions, with an independent verifier. A pull request with unmet criteria lists them and the issue stays open. (source: `docs/issue-workflow.md` rule 6 and section 5; user, 2026-10-20)
17. A harness finding that is a production bug gets its own issue. No drive-by fix. A flaky test gets its own issue; never retry silently. (source: user, 2026-10-30 "Denne må få et eget issue")
18. Data is spot only and long only. The default financial feature set is the full expanded set. Downloads stay outside `rcsr`. (source: user, 2026-10-22)

## Models and agents

19. Sonnet implements, rebases, salvages and verifies by default. Opus takes hard and demanding tasks (research on unclear issues and genuinely difficult implementation), always with tractatus thinking and context7, never web search or web fetch. Set `model` on every dispatch. (source: user, 2026-10-23; one recorded exception: #352, 2026-10-30; user, 2026-10-31)
20. Every brief starts with: "You are not resumed automatically; block in your own turn until everything is done." (source: global orchestration rules)
21. At most 10 concurrent agents, file-disjoint, one worktree and one dev environment each, unique ports. Code for many tickets is written in parallel; only builds and tests serialise through the lock. At most 3 tickets use the machine at once; a ticket whose agent waits on the lock does not count, so start another. (source: global orchestration rules; 2026-10-30 exception for #352; user, 2026-10-31)
22. On a new finding (bug, regression, warning) dispatch investigation at once and report afterwards. (source: global orchestration rules)

## Machine and waiting

23. Builds (`cargo build`, `cargo test --no-run`) run through the build slot `/mnt/lager/tmp/opencode/build-slot.sh`, `CARGO_BUILD_JOBS=2`, up to 3 concurrent machine-wide at nice 19 and ionice idle, only while MemAvailable is at least 8 GiB and the PAUSE file is absent. Test runs and `rcsr-bench` run through the single test lock `/mnt/lager/tmp/opencode/one-full-suite.sh`, one at a time. Each hold is under 10 minutes, with an outer `timeout` on anything that can hang. Ticket tests are minimal: only the tests a change adds or touches. (source: `docs/issue-workflow.md` section 6; user, 2026-10-18 "only one full-suite test at a time"; user, 2026-10-31)
24. The one full test (gate plus large-store arms) runs once on the integration tip, after all issues are done. Never one per PR or per blocker. (source: user, 2026-10-30; user, 2026-10-31 "kun skal være en eneste full test når alle issues er ferdig")
25. The desktop has priority. Honour the `PAUSE` file. Never run the 39.8 GB test. Do not work around the lock or nice. (source: user, 2026-10-30)
26. No `pgrep -f` or `pkill -f` loops: they match their own command line and never end. No `pkill`, no `killall`. Kill only a pid you started. (source: failures F1, F2, F6; user, 2026-10-30)
27. One wait mechanism per agent: one blocking call, or one background shell and its notification. Never repeated wait loops. (source: failure F2; user, 2026-10-30)
28. A watchdog watches the newest file mtime of each agent worktree and alerts after 45 minutes of silence. Transcripts buffer and are not a liveness signal. (source: global orchestration rules)

## Runs

29. A long run starts only when its preconditions hold, and then at once. GPU by default with an explicit backend. Baselines beside every result. (source: user, 2026-10-20 and 2026-10-21)
30. `HANDOFF.md` is present tense, untracked and temporary. (source: user, 2026-10-22)

## GPU

31. A device out-of-bounds access can crash the NVIDIA driver and freeze the whole machine. Any test that changes CUDA code, squeezes device memory or exercises an allocation-failure path runs first under `/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --leak-check no --error-exitcode 99` (test binary built with `cargo test --no-run`, small inputs, outer `timeout`, through the lock). No plain GPU run of changed CUDA code until that run is clean. A refused device allocation leaves no dangling device pointer and no pending CUDA error. After GPU work `journalctl -k` shows no new `Xid` line. The system `compute-sanitizer` 2022.4.1 (CUDA 12.0) cannot attach on driver 580, and memcheck does not scale to the 16.1M-row store (use it on small tests). (source: user, 2026-10-31; machine freeze, #381)
