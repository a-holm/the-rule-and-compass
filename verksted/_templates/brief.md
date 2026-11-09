# Brief ticket section: <issue number and title>

Hard rules: never `pkill` or `killall` (kill only a pid you started, with `kill <pid>`); a wait uses `wait-for.sh`, never `pgrep -f`; no Python. A lint in `verksted_tools` fails on `pkill` and `killall` in scripts.

Render the dispatch prompt with `../_tools/brief.sh impl --issue <N> --slug <slug>`; it names `../_shared/brief.md` and this file. Save it as `../03_orchestrate/stages/02_brief/output/brief-<N>.md` and append the filled ticket fields below. Replace every angle-bracket placeholder in what you append.

Ticket fields (copy these lines into the brief; the steps stay here):
Ticket: #<N>, slug <slug>, epic #<epic>, model <sonnet|opus>.
Worktree: <absolute path, for example the repo root plus /worktrees/issue-N-slug>
Base: <integration branch>; branch issue/<N>-<slug>.
Owned files: <paths this agent may change>
Forbidden files: <paths other agents own>
Design: <link to the decision comment, or "the issue body is the design">
Acceptance criteria: <copy the checkbox list>
Not to touch: parked issues and everything outside the owned files.

Steps:
1. Read the issue body and every comment: `gh issue view <N> --comments`. The body is the spec. If it links research notes (`arbeidsmappe/crypto-rcsr/research/<slug>/decision.md`), read them too. If a choice is not covered, or a criterion fails twice for the same reason, stop and report the exact question.
2. Setup from the repo root: `verksted/_tools/wt-new.sh <N> <slug>`. It fetches origin and creates the worktree and branch from the base. Work only there. If the base moves, `git merge origin/<base>`; never rebase after pushing, never force-push.
3. Read and follow every line of `../_shared/brief.md`: builds only through `rc-run.sh build`, tests only through `rc-run.sh test`, minimal tests by exact filter, prove red first with a WIP commit, no Python ever, no git stash, no pkill. A wait uses `verksted/_tools/wait-for.sh`, never `pgrep -f`.
4. The integration branch has a zero-warning baseline (CPU and CUDA builds) and `tests/test_attributes.rs` guards lost `#[test]` attributes. Add no warnings; run `--test test_attributes` and `--test sanitizer_plan` once at the end.
5. GPU: any `.cu` change, allocation-failure path or device test of changed code runs first under `verksted/_tools/sanitize.sh` (small inputs, through the lock) before any plain GPU run. Search-level GPU tests set `dispatch: DispatchMode::Device`. Use `rc_checked_malloc` and `RC_CUB`. The Xid count must not rise; stop if it does.
6. Many agents edit `src/search.rs`, `src/main.rs`, `src/schema.rs` and `tests/gpu_fit.rs` in parallel: keep hunks small and local.
7. Update `--help`, `rcsr schema`, README and `docs/` when behaviour, flags or output change. Measured numbers go in the PR body, never in code comments.
8. Open the PR against the base with `Refs #<N>, #<epic>` (epic from the issue metadata; #281 if none). Do not merge.
9. Final report: PR number, head sha, each acceptance criterion with test name or log path, deviations with reasons, open items, and any finding outside the issue (report it, do not fix it).
