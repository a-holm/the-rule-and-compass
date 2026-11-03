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

9. No Python. No `.py` file, no `python3` call, no python shebang. Experiments, reference values, fixtures and helpers are Rust: crate tests, or single-file `rustc -O` programs outside the repo. Quick checks are shell. Scripted edits use the Edit tool, `perl` or `sed`. A Claude Code PreToolUse hook (`~/.claude/hooks/eureqa-no-python.sh`) blocks any Bash command that invokes `python` or `pip` in this repository. (source: `docs/issue-workflow.md` rule 10; `my-version/tests/no_python.rs`; user, 2026-10-31)
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

23. Builds (`cargo build`, `cargo test --no-run`) run through the build slot `/mnt/lager/tmp/opencode/build-slot.sh`, `CARGO_BUILD_JOBS=2`, up to 3 concurrent machine-wide (1 while the desktop streams), only while MemAvailable is at least 8 GiB and the PAUSE file is absent. Both scripts run the command under SCHED_IDLE (`chrt -i 0`), nice 19 and ionice idle. While the desktop streams video (`desktop-busy.sh`: GPU video decoder or encoder busy, or the file `/mnt/lager/tmp/opencode/STREAMING` exists) both pin work to CPUs 2,3,6,7 and set `RUST_TEST_THREADS=4`. Test runs and `rcsr-bench` run through the single test lock `/mnt/lager/tmp/opencode/one-full-suite.sh`, one at a time. Each hold is under 10 minutes, with an outer `timeout` on anything that can hang. Ticket tests are minimal: only the tests a change adds or touches. (source: `docs/issue-workflow.md` section 6; user, 2026-10-18 "only one full-suite test at a time"; user, 2026-10-31)
24. The one full test (gate plus large-store arms) runs once on the integration tip, after all issues are done. Never one per PR or per blocker. (source: user, 2026-10-30; user, 2026-10-31 "kun skal være en eneste full test når alle issues er ferdig")
25. The desktop has priority. Honour the `PAUSE` file. Never run the 39.8 GB test. Do not work around the lock, SCHED_IDLE, nice or the core pinning. The desktop gets all the CPU it wants. While it streams video, cores 0 and 1 and their hyperthreads stay free. (source: user, 2026-10-30; user, 2026-10-31)
26. No `pgrep -f` or `pkill -f` loops: they match their own command line and never end. No `pkill`, no `killall`. Kill only a pid you started. (source: failures F1, F2, F6; user, 2026-10-30)
27. One wait mechanism per agent: one blocking call, or one background shell and its notification. Never repeated wait loops. (source: failure F2; user, 2026-10-30)
28. A watchdog watches the newest file mtime of each agent worktree and alerts after 45 minutes of silence. Transcripts buffer and are not a liveness signal. (source: global orchestration rules)

## Runs

29. A long run starts only when its preconditions hold, and then at once. GPU by default with an explicit backend. Baselines beside every result. (source: user, 2026-10-20 and 2026-10-21)
30. `HANDOFF.md` is present tense, untracked and temporary. (source: user, 2026-10-22)

## GPU

31. A device out-of-bounds access can crash the NVIDIA driver and freeze the whole machine. Any test that changes CUDA code, squeezes device memory or exercises an allocation-failure path runs first under `/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --leak-check no --error-exitcode 99` (test binary built with `cargo test --no-run`, small inputs, outer `timeout`, through the lock). Search-level tests need `--dispatch device` (CLI) or a test config with `dispatch: DispatchMode::Device`: the sanitized device is slow, so `auto` routes every stage to the CPU, no kernel launches and a clean memcheck proves nothing. No plain GPU run of changed CUDA code until that run is clean. A refused device allocation leaves no dangling device pointer and no pending CUDA error. After GPU work `journalctl -k` shows no new `Xid` line. The system `compute-sanitizer` 2022.4.1 (CUDA 12.0) cannot attach on driver 580, and memcheck does not scale to the 16.1M-row store (use it on small tests). (source: user, 2026-10-31; machine freeze, #381)

## Git and evidence

32. Never `git stash`: the stash is shared by every worktree. Prove red with a temporary WIP commit. (source: failure F30; `common-rules` brief, 2026-11-01)
33. Never rewrite your branch history: no `reset --soft`, no rebase, no amend after push, no squash. Two agents reverted base files that way. Merge the base branch; the squash happens at merge time. (source: failure F31; 2026-11-01)
34. Every build or test result you report has a log under `/mnt/lager/tmp/rcsr-gates/`. A result without a log counts as not run. After merging the base, rerun and log the tests you report. (source: failure F33; 2026-11-01)
35. Issue comments, PR bodies and commit bodies state only facts that are true now and verified (log path, test name, `git`, `gh`). No predictions. Write "not run", "open" or "not verified". Attribute any claim you did not check. Posting issue comments is always allowed. (source: failure F34; user, 2026-11-01)
36. A change that adds a refusal or check on a shared path greps `tests/` for every test of that path and runs those files too. (source: failure F32; 2026-11-01)
37. Leave no long background job (nohup, multi-hour loops) after the final report: it holds the shared lock. Long A/B or bench sweeps belong to the final measurement phase. (source: failure F36; 2026-11-01)

## Measurement

38. Wall time under SCHED_IDLE at high load is noise. Compare A and B inside one lock hold, alternate the arm order, report user CPU time, and never compare with older timings. (source: failure F35; 2026-11-01)

## Orchestration

39. The orchestrator reads the code behind every deviation, workaround or odd number in a hand-back and files real defects as issues at once. An agent that judges a task impossible or too hard reports the exact blocker with evidence; an Opus agent with tractatus thinking and context7 double-checks before the verdict is accepted. (source: user, 2026-10-31 and 2026-11-01)
40. Delegate everything that can be delegated; the main session only orchestrates, decides and merges, with the repo root as its folder. Plan and synthesise with Opus (tractatus thinking and context7, no web search or web fetch); execute with Sonnet. An Opus agent that stops on an API error is rerun on Sonnet. (source: user, 2026-11-01 and 2026-11-02)
41. Merges into the integration branch after a merge clerk verified the evidence are allowed. Nothing merges into `main` without the project lead's explicit instruction. After each merge, comment on the issue with the merge sha and verified facts. (source: user, 2026-10-31 and 2026-11-01)

## Desktop and hygiene

42. Cinnamon screen blanking and screen lock stay off, so the user can watch the machine. (source: user, 2026-10-31)
43. No mention of the origin product or of reverse engineering in code, docs or git history. Machine paths use neutral names. (source: user, 2026-10-18)

## User rules, verbatim

Quotes are copied as typed, typos included, in Norwegian or English. Names are elided in square brackets.

| Date | Quote, as typed | Rule |
|---|---|---|
| 2026-10-18 | only one full-suite test at a time. | 23 |
| 2026-10-18 | Jeg vil ikke ha en eneste omtale av [origin product] i koden eller git-historikken | 43 |
| 2026-10-19 | remember exactness is necessary, anything that leads to less exactness is not an optimization | 14 |
| 2026-10-20 | stop making python script evaluators, make highly optimized rust applications that can be reused instead | 9 |
| 2026-10-20 | helt latterlig å lukke en issue uten å implementere alle deler | 16 |
| 2026-10-20 | by default it should run with gpu | 29 |
| 2026-10-20 | clean up any development nonsense, old worktrees and branches | 7 |
| 2026-10-21 | jeg forbyr Closes messages unntatt når jeg lukker | 4 |
| 2026-10-21 | du må BARE kjøre det som subagents med egen worktree og egen branch, arbeid aldri med issues på en annen måte | 1 and 2 |
| 2026-10-22 | Husk vi vil ha mest mulig GPU bruk | 29 |
| 2026-10-22 | kun long posisjoner av religiøse grunner | 18 |
| 2026-10-22 | Alle features skal finnes der, til vi har bevist matematisk at det aldri i noen tilfeller er til nytte | 18 |
| 2026-10-22 | pass på tester, du tester en million ting samtidig | 23 |
| 2026-10-30 | jeg prøver å se stream på discord, det må prioriteres | 25 |
| 2026-10-30 | hvorfor kjører du gaten så mange ganger. Bare merge dem til integration worktree og så restarter jeg og såp kan du arbeide mot integration worktreet og kjøre gaten en gang til slutt for alle issues | 24 |
| 2026-10-30 | backtrace burde alltid vært tilgjengelig under utvikling vel | F18 |
| 2026-10-31 | kjøær experimenter i rust siden det skal programmeres i rust. experimenter i python gir ikke mening | 9 |
| 2026-10-31 | ingen python script skal eksistere, alt skal være rust | 9 |
| 2026-10-31 | det bør være en regel i verkstedet | 9 |
| 2026-10-31 | vær snill å stopp all sånn "skru av skjerm" og "lås skjerm" greier på cinammon slik at jeg kan se hva som foregår selv om jeg går vekk fra skjermen en stund. | 42 |
| 2026-10-31 | den frøs ikke på grunn av driver, men på grunn av dårlig programmering | 31 |
| 2026-10-31 | du kan likevel pårogrammere 1000000000000 issues uten å teste. og du kjører ikke engang de maks 3 tickets heller. og maks 3 gjelder ikke hvis de venter på locken da kan du starte ny | 21 |
| 2026-10-31 | alle tester skal være minimale og inn i integration branch der det kun skal være en eneste full test når alle issues er ferdig | 24 |
| 2026-10-31 | opus er greit for vanskelige og krevende oppgaver, men kun med tractatus_thinking og context7 | 19 |
| 2026-10-31 | oppdater hver issue med en kommentar når de er merget inn i integrasjons-branchen om det er nyttig | 41 |
| 2026-10-31 | discord må ha 100% av cpu som den vil ha | 25 |
| 2026-10-31 | veldig bra at du finner slike ting | 39 |
| 2026-11-01 | man skal alltid ha lov til å poste kommentarer på issues | 35 |
| 2026-11-01 | ikke skriv kommentarer som er usanne eller som kan være usanne. | 35 |
| 2026-11-01 | det betyr at du ikke skal forutsi fremtiden og lignende | 35 |
| 2026-11-01 | regelen er, kan du delegere det til en underagent så gjør det. enten med opus med tractatus_thinking og context7 om det er et vanskelig spørsmål eller veldig kritisk. eller sonnet hvis ikke. | 40 |
| 2026-11-01 | hvis en agent bedømmer noe som umulig eller for vanskelig slik at den feiler jobben så bruk opus agenter med tractatus_thinking og context7 og dyp research for å dobbeltsjekke. | 39 |
| 2026-11-01 | pilot v2 må være stor og grundig slik at man virkelig får kjørt | 29 |
| 2026-11-02 | aldri arbeide direkte, kun med subagents. [repo root] skal være mappen din | 40 |
| 2026-11-02 | burde bruke 13 hvis den kan | 31 |
