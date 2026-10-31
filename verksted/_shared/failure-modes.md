# Failure modes seen in the logs, each with its guard

Mined from the Claude Code and opencode sessions of this repo (2026-10-18 to 2026-10-31). The guard is the rule in `rules.md` and the stage check that catches it.

| # | Failure | Evidence | Guard |
|---|---|---|---|
| F1 | Self-matching `pgrep -f` wait loops never end; 15 zombie loops, 12 of them 6 to 14 hours old | orchestrator session 2026-10-30 21:07 | rule 26; `../03_orchestrate/stages/04_watchdog/CONTEXT.md` |
| F2 | 57 duplicate wait loops from one agent, 80 active shells | same session, 21:08 | rule 27; brief |
| F3 | Lock held for hours by a hung test (#303 held it 90+ minutes; a 30-minute `rcsr-bench` blocked all) | 2026-10-30 20:42, opencode 2026-10-20 | rule 23; `../05_diagnose/CONTEXT.md` |
| F4 | Lock bypassed by a direct `cargo` call | opencode 2026-10-20 20:27; user "where did the lock go" | rule 23; `../10_machine/stages/01_lock/CONTEXT.md` |
| F5 | Force-push refused as destructive | classifier denial 2026-10-30 07:03 | rule 3 |
| F6 | `pkill` by agents; a `pkill -f` killed the agent's own shell | seven calls, denied | rule 26 |
| F7 | Stale claim taken from an agent's report: "probably fitted on the test block", inferred not checked; three gaps behind a green PR (#257) | verifier on #257 | rule 13; `../08_close/stages/01_verifier/CONTEXT.md` |
| F8 | Alerts from agents that had already finished | 2026-10-30 21:08 | agents leave wait loops before the final report; rule 27 |
| F9 | Python experiments and `python3 -` edits (317 shell lines) | user 2026-10-31 | rule 9; `tests/no_python.rs` |
| F10 | One full gate per PR (20 gates, about an hour each) | user 2026-10-30 22:58 | rule 24; `../04_gate/CONTEXT.md` |
| F11 | `Closes` in pushed commits closed issues | opencode 2026-10-21 | rule 4; `../01_issue/stages/04_implement/CONTEXT.md` |
| F12 | Issue closed without every part implemented, or work deferred | opencode 2026-10-20 | rule 16; `../08_close/CONTEXT.md` |
| F13 | Wrong assumption found only after a 10-hour run: the gain was the label's own normalisation; 95.4 % of wall time was a serial CPU fallback | `run-audit/10h` synthesis | rules 12, 29; `../07_longrun/stages/05_audit/CONTEXT.md` |
| F14 | Premises corrected by measurement: #364 locality test already passed; #352 cause proved a cycle; `ou_hl96` tolerance 1e-12 impossible | 2026-10-30 and 2026-10-31 | rule 12; `../02_research/stages/03_premise/CONTEXT.md` |
| F15 | Eight `rcsr` processes where one was meant | opencode 2026-10-20 15:09 | `../07_longrun/stages/01_preconditions/CONTEXT.md` |
| F16 | Run started before required work finished, or idle waiting | opencode 2026-10-20 and 2026-10-21 | rule 29 |
| F17 | Desktop starved by a terminal in software rendering (345 % CPU) | 2026-10-30 21:41 | `../10_machine/stages/03_measure/CONTEXT.md` |
| F18 | GPU test could not be traced (`ptrace_scope=1`); no stack | 2026-10-30 20:44 | `RCSR_ATTACHABLE=1`; `../05_diagnose/stages/02_stack/CONTEXT.md` |
| F19 | Flaky test failed the gate (a timing comparison; a 3.5e-9 relative miss once in 22 runs) | #311, #359 | rule 17 |
| F20 | Four tickets active against a cap of three | 2026-10-30 | rule 21; `../03_orchestrate/stages/01_wave_plan/CONTEXT.md` |
| F21 | Dead agent left unreviewed work in its worktree | #295, #301 salvage | `../03_orchestrate/stages/03_dispatch/CONTEXT.md`: a new Sonnet agent treats it as a draft |
| F22 | Shared `CARGO_TARGET_DIR` linked artefacts from another tree | opencode 2026-10-18, twice | rule 6 |
| F23 | DNS failure during fetch or merge; retry loops | 2026-10-30 | verify with `gh pr view --json state,mergeCommit` after a retry |
| F24 | `rm -rf` with variables denied by a safety check | three agents | rule 7: `git worktree remove` with an explicit path |
| F25 | Device out-of-bounds read after a refused allocation (Xid 31); nvidia_uvm NULL dereference; every CPU soft-locked; desktop frozen 07:33 to 13:07 | kernel log 2026-10-31, #381 | rule 31 |
