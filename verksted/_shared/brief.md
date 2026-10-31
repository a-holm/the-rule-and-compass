You are not resumed automatically; block in your own turn until everything is done.

Rules for every agent in this repo. The ticket section follows below.

- Read `AGENTS.md` and `docs/issue-workflow.md` first. Work only in the worktree named below. Never touch the repo root or another worktree.
- Open your pull request against the base branch named below. Do not merge; the orchestrator merges.
- Run only the tests your change needs: prove red first, then green. Never run the full suite; the orchestrator runs one gate on the integration branch.
- Every cargo command goes through the machine lock, with `CARGO_BUILD_JOBS=2`, and logs to `/mnt/lager/tmp/eureqa-gates/issue$N-*.log`:
  `L=/mnt/lager/tmp/opencode/one-full-suite.sh`
  `$L bash -c "cd $W/my-version && CARGO_BUILD_JOBS=2 RCSR_CUDA_ARCH=sm_86 timeout 580 cargo test --release --features cuda --test FILE FILTER" > /mnt/lager/tmp/eureqa-gates/issue$N-dev1.log 2>&1`
  The lock runs at nice 19 and ionice idle and waits while `/mnt/lager/tmp/opencode/PAUSE` exists. Do not work around either. The user's desktop has priority.
- Never run two cargo processes at once, or cargo outside the lock. Keep each lock hold under 10 minutes. Give anything that can hang an outer `timeout`.
- Waiting: one blocking call, or one background shell and its notification. Never start repeated wait loops. Never write `pgrep -f` or `until ! pgrep -f` loops: they match their own command line and never end.
- Never use `pkill` or `killall`. Kill only a process you started.
- No Python, ever: no `python3`, no `.py` file, no Python fixtures or golden values. Experiments, reference computations and helpers are Rust: crate tests, or single-file programs built with `nice -n 19 rustc -O` outside the repo. Quick checks are shell.
- GPU safety: changed CUDA code, squeezed device memory and allocation-failure tests run first under `compute-sanitizer --tool memcheck --leak-check no --error-exitcode 99`, through the lock, with an outer `timeout`. No plain GPU run until it is clean. After GPU work `journalctl -k` shows no new `Xid` line (rule 31).
- Git: `git commit -m` with a conventional subject (`fix(area): ...`, `feat(area): ...`, `test(area): ...`) and the body line `Refs #N`. Never `Closes` or `Fixes`. No AI attribution, no co-author trailers. Never force-push. Never rebase after a push: merge the base branch instead.
- Style: plain technical English, no em-dashes. Code comments only for constraints the code cannot show; no measured numbers in comments. Update `--help`, `rcsr schema`, `README.md` and `docs/` when behaviour changes.
- If the issue does not cover a choice, or a criterion fails twice for the same reason, stop and report the exact question.
- PR body: what changed, prove-red and green output (commands and counts), deviations with reasons, `Refs #N`.

Final report: PR number, head sha, each acceptance criterion with its test name or log path, what you did not verify, deviations, open items.
