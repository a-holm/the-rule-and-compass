---
type: reference
description: Numbered hard requirements for every pipeline, each with its source.
---

# Rules: hard requirements for every pipeline

The rules most often broken, each with its source. Sources: `AGENTS.md` and `docs/issue-workflow.md` of this repo, the user's global rules, and dated user instructions ("user, YYYY-MM-DD"). Failures behind them: `failure-modes.md`.

## Work surface

1. One issue, one worktree, one branch, one pull request. Branch `issue/<n>-<slug>`, worktree `worktrees/issue-<n>-<slug>/`. (source: `AGENTS.md` Non-negotiables; `docs/issue-workflow.md` rules 2 to 4)
2. Work only in your own worktree. Never touch the repo root or another agent's worktree. The orchestrator delegates fixes back to the agent or dispatches a new one. (source: `AGENTS.md`; user, 2026-10-21; global orchestration rules)
3. Never commit to `main`. Never force-push. Never rebase after a push: merge the base branch into yours. (source: `AGENTS.md`; the 2026-10-30 refusal of a `--force-with-lease` brief)
4. `Refs #<n>`, never `Closes`, `Fixes` or `Resolves`. (source: `AGENTS.md` Style; `docs/issue-workflow.md` rule 7; user, 2026-10-21)
5. Commits follow the repo style `feat(area): ...`, `fix(area): ...`, `perf(area): ...`, `docs: ...` with a `Refs #<n>` body line. This repo style replaces the global `#<n>: ` subject prefix. Squash subjects are hand-written: `type(area): subject (#<pr>)`. No AI attribution, no co-author trailers. (source: `AGENTS.md` Style; global commit rules)
6. Each worktree keeps its own `target/` in the crate directory (`verksted/_tools/crate-dir.sh "$W"`). Never share a `CARGO_TARGET_DIR`. (source: `docs/issue-workflow.md` sections 2 and 6; stale artefacts seen twice, 2026-10-18)
7. Clean up path by path, only the finished ticket's worktree and branches. No `rm -rf`, no bulk clean-up while agents live. Merge conditions: rule 62. (source: global orchestration rules; denied removals, 2026-10-30)
8. Keep scratch, datasets and build output out of git. (source: `AGENTS.md` Non-negotiables)

## Language and tools

9. No Python. No `.py` file, no `python3` call, no python shebang. Experiments, reference values, fixtures and helpers are Rust: crate tests, or single-file `rustc -O` programs outside the repo. Quick checks are shell. Scripted edits use the Edit tool, `perl` or `sed`. A Claude Code PreToolUse hook in the user settings blocks any Bash command that invokes `python` or `pip` in this repository. (source: `docs/issue-workflow.md` rule 10; the engine crate's `tests/no_python.rs`; user, 2026-10-31)
10. Plain technical English, short sentences, no em-dashes. Docs describe the present only: no history, no provenance, no AI references. (source: global writing rules)
11. Code comments state a constraint the code cannot show. No measured numbers in comments. (source: global commit and ticket rules)

## Quality

12. Prove red first, then green: the new test fails on the unfixed code. Measure before building; label every claim measured, derived or inferred. (source: `AGENTS.md` "Prefer a measurement or a test over an assertion"; user, 2026-10-30 "check, never guess")
13. Verify against PR state, `git log` and log files, never an agent's self-report. An inferred claim is not evidence. (source: global orchestration rules)
14. Reported values are exact CPU f64. Never weaken an exactness guard. Never claim GPU work that did not execute. (source: user, 2026-10-18 and 2026-10-19 "exactness is a goal")
15. Update `--help`, `rcsr schema`, `README.md` and `docs/` when behaviour, flags or output change. (source: `AGENTS.md` Non-negotiables)
16. An issue closes only after an independent verifier mapped every acceptance criterion to evidence at the merge (`08_close/stages/01_verifier`). A criterion that needs a GPU run, the full suite or a measurement and was not run is a `not run` row; rule 58 decides whether the issue stays open. A pull request with unmet criteria lists them. (source: `docs/issue-workflow.md` rule 6 and section 5; user, 2026-10-20; user, 2026-11-22)
17. A harness finding that is a production bug gets its own issue. No drive-by fix. A flaky test is a defect: rule 57. (source: user, 2026-10-30 "Denne må få et eget issue")
18. Data is spot only and long only. No short, margin or options. The default financial feature set is the full expanded set. Downloads stay outside `rcsr`. (source: user, 2026-10-22)

## Models and agents

19. Model tiers: `models.md`. Set the model or agent type on every dispatch. (source: user, 2026-10-23, 2026-10-31 and 2026-11-22; one recorded exception: #352, 2026-10-30)
20. Every brief starts with: "You are not resumed automatically; block in your own turn until everything is done." (source: global orchestration rules)
21. At most 10 concurrent agents, file-disjoint, one worktree and one dev environment each, unique ports. Code for many tickets is written in parallel; only builds and tests serialise through the lock. At most 3 tickets use the machine at once; a ticket whose agent waits on the lock does not count, so start another. (source: global orchestration rules; 2026-10-30 exception for #352; user, 2026-10-31)
22. On a new finding (bug, regression, warning) dispatch investigation at once and report afterwards. (source: global orchestration rules)

## Machine and waiting

23. Builds (`cargo build`, `cargo test --no-run`) run through `BUILD_SLOT` with `BUILD_JOBS` jobs, at most `BUILD_SLOTS` at once (`BUILD_SLOTS_BUSY` while the desktop is busy), only while MemAvailable is at least `MEM_MIN_FREE` and the `PAUSE` file is absent. Test runs and benchmark runs go through the single `TEST_LOCK`, one at a time; each hold is under 10 minutes, with an outer `timeout` on anything that can hang. Both scripts run the command under SCHED_IDLE, nice 19 and ionice idle and pin it to `PIN_CORES_BUILD` (`PIN_CORES_BUSY` while the desktop is busy). Ticket tests are minimal: only the tests a change adds or touches. Values: `machine.md`. (source: `docs/issue-workflow.md` section 6; user, 2026-10-18 "only one full-suite test at a time"; user, 2026-10-31)
24. The one full test (gate plus large-store arms) runs once on the integration tip, after all issues are done. Never one per PR or per blocker. (source: user, 2026-10-30; user, 2026-10-31 "kun skal være en eneste full test når alle issues er ferdig")
25. The desktop has priority. Honour the `PAUSE` file. Never run the large-store tests that `rules-local.md` names. Do not work around the lock, SCHED_IDLE, nice or the core pinning. The desktop gets all the CPU it wants; while it is busy, `CORES_RESERVED` stay free. (source: user, 2026-10-30; user, 2026-10-31)
26. No `pgrep -f` or `pkill -f` loops: they match their own command line and never end. No `pkill`, no `killall`. Kill only a pid you started. (source: failures F1, F2, F6; user, 2026-10-30)
27. One wait mechanism per agent: one blocking call, or one background shell and its notification. Never repeated wait loops. (source: failure F2; user, 2026-10-30)
28. A watchdog watches the newest file mtime of each agent worktree and alerts after 45 minutes of silence. Transcripts buffer and are not a liveness signal. (source: global orchestration rules)

## Runs

29. A long run starts only when its preconditions hold, and then at once. GPU by default with an explicit backend. Baselines beside every result. (source: user, 2026-10-20 and 2026-10-21)
30. `HANDOFF.md` is present tense, untracked and temporary. (source: user, 2026-10-22)

## GPU

31. A device out-of-bounds access can crash the NVIDIA driver and freeze the whole machine. Any test that changes CUDA code, squeezes device memory or exercises an allocation-failure path runs first under `$SANITIZER` (`machine.md`) with `--tool memcheck --leak-check no --error-exitcode 99` (test binary built with `cargo test --no-run`, small inputs, outer `timeout`, through the lock). Search-level tests need `--dispatch device` (CLI) or a test config with `dispatch: DispatchMode::Device`: the sanitized device is slow, so `auto` routes every stage to the CPU, no kernel launches and a clean memcheck proves nothing. No plain GPU run of changed CUDA code until that run is clean. A refused device allocation leaves no dangling device pointer and no pending CUDA error. Memcheck runs on small tests only: it does not scale to large stores. Before a GPU step record the time; after it `journalctl -k --since <time> | grep -c 'NVRM: Xid'` must be 0. Old lines rotate out of the journal, so a total is never compared and a falling total is not a fault. A new line stops all GPU work; report it. (source: user, 2026-10-31; machine freeze, #381; Xid by new lines, 2026-11-20)

## Git and evidence

32. Never `git stash`: the stash is shared by every worktree. Prove red with a temporary WIP commit. (source: failure F30; `common-rules` brief, 2026-11-01)
33. Never rewrite your branch history: no `reset --soft`, no rebase, no amend after push, no squash. Two agents reverted base files that way. Merge the base branch; the squash happens at merge time. (source: failure F31; 2026-11-01)
34. Every build or test result you report has a log under `$GATES` (`lib.sh`). A result without a log counts as not run. After merging the base, rerun and log the tests you report. (source: failure F33; 2026-11-01)
35. Issue comments, PR bodies and commit bodies state only facts that are true now and verified (log path, test name, `git`, `gh`). No predictions. Write "not run", "open" or "not verified". Attribute any claim you did not check. Posting issue comments is always allowed. (source: failure F34; user, 2026-11-01)
36. A change that adds a refusal or check on a shared path greps `tests/` for every test of that path and runs those files too. (source: failure F32; 2026-11-01)
37. Leave no long background job (nohup, multi-hour loops) after the final report: it holds the shared lock. Long A/B or bench sweeps belong to the final measurement phase. (source: failure F36; 2026-11-01)

## Measurement

38. Wall time under SCHED_IDLE at high load is noise. Compare A and B inside one lock hold, alternate the arm order, report user CPU time, and never compare with older timings. (source: failure F35; 2026-11-01)

## Orchestration

39. The orchestrator reads the code behind every deviation, workaround or odd number in a hand-back and files real defects as issues at once. An agent that judges a task impossible or too hard reports the exact blocker with evidence; an Opus agent double-checks before the verdict is accepted (`models.md`). (source: user, 2026-10-31 and 2026-11-01)
40. Delegate everything that can be delegated; the main session only orchestrates, decides and merges, with the repo root as its folder. Model tiers: `models.md`. (source: user, 2026-11-01, 2026-11-02 and 2026-11-22)
41. Merges into the integration branch after a verifier verified the evidence are allowed. Nothing merges into `main` without the project lead's explicit instruction. After each merge, comment on the issue with the merge sha and verified facts. (source: user, 2026-10-31 and 2026-11-01)

## Desktop and hygiene

42. Cinnamon screen blanking and screen lock stay off, so the user can watch the machine. (source: user, 2026-10-31)
43. No mention of the origin product or of reverse engineering in code, docs or git history. Machine paths use neutral names. GitHub text names log paths through `$GATES`. Run `leak-scan.sh --diff` before a PR and on any body you post; the banned-words list holds the old machine names. (source: user, 2026-10-18)

## Data

44. Maximum history: every store, run and window starts at the first bar of its source; a later start needs the project lead's approval. (source: user, 2026-11-14)
45. Benchmarks: Sharpe, Sortino and every excess-return figure use cash at 0, buy-and-hold or the equal-weight universe as the benchmark; never an interest-bearing rate or yield. (source: user, 2026-11-14)
46. Locked ranges: declared with the data; download and checksum only; never open, print or summarise a locked row; read once through seal and sealed read; sidecars of a locked store are read only with filters whose output is key names or counts. Filters print key names, counts, or values first filtered to lie outside the range. Never write a symbol, date or value from inside a locked range into GitHub text. Dates and store names of the current locks: `rules-local.md`. (source: 2026-11-14)
47. Data licence: research-only data and raw vendor files never enter git; the dataset card records licence and attribution. (source: `AGENTS.md` Non-negotiables; the terms of a public vendor data archive, read 2026-11-22, forbid commercial use without an agreement)

## Tools and hooks

48. Never bypass a git hook. A hook that blocks a red WIP commit: log the red run instead, or ask the orchestrator. (source: `common-rules` brief, 2026-11)
49. The banned interpreter's name is never typed in a shell command, also not in a grep pattern or a commit message typed on a command line. (source: 2026-11-16)
50. Never `--edit-last` on an issue or PR comment: all agents share one account. Edit a comment by its id. Every `gh` call names the repo unless the cwd is a checkout. Before `--body-file F`: `test -s F && grep -q '^## ' F`. (source: 2026-11-22)
51. CPU load generators start and stop inside the command the lock runs. A script test never signals a process outside its temp tree and never names a real lock path. Every lock path in a test comes from an override that points into the test's temp dir. A tool that reaps by lock holder acts only on the lock its own process holds. (source: `common-rules` brief, 2026-11)

## Waiting and builds

52. One tracked wait per job: one background `wait-for.sh` call that ends only at the job's done line or file, with a timeout as long as the job. Never a series of short waits: each one posts a notice to the project lead. Before starting a wait, check that none of yours watches the same target; `wait-for.sh` refuses a second wait on one target. Never detach a slot call and hand back; hand back only with no background job of yours running. (source: 2026-11-15, 2026-11-21; user, 2026-11-16 and 2026-11-22)
53. Queue a build-slot job only when you need its result and do not have it yet: never a second copy of a job that is queued, running or finished (`rc-run.sh` refuses a duplicate). One job per purpose; batch your test files into one job. Exit 75 means the slot gave up waiting (`BUILD_SLOT_WAIT_MAX`), not a build failure: requeue the same job once and report a second exit 75. (source: 2026-11-22)
54. One `rcsr` binary per feature set. Test executables spawn the `rcsr` binary of their worktree's release target, which the last build wrote. A CPU build after a CUDA build replaces it with a binary without the CUDA hooks, so CUDA tests that spawn it (fault injection, device CLI tests) run the wrong binary and fail or pass for the wrong reason. Make the build that matches the tests the last one before running them, or run them through `cargo test` with the same features. `sanitize.sh` and `rc-run.sh run` refuse a mismatch. (source: 2026-11-22)

## Evidence

55. The final report pastes `git diff --stat <base>...HEAD`; every file claimed appears there. (source: 2026-11-15)
56. A change to a hashed byte format runs every golden and pinned-hash test before the merge, not only its own tests. Hashed byte formats: store sidecars, schema and registry records, lock records, report hash inputs, and every constant that feeds one. Pinned-hash tests: every test file whose name contains `golden`, and every test file that pins a sha256 or a view id (`verify-merged.sh --goldens` lists them). (source: 2026-11-21; #1294 left a lock-record golden test red on the tip)
57. A flaky test is a defect: a test that can pass or fail on the same code. Fix the cause. Never call a failure a flake and move on, never rerun until green. Tests assert behaviour or counted work, never elapsed time; each test uses its own temp paths; real non-determinism is fixed, not tolerated. (source: user, 2026-11-03)
58. A merged issue is verified by its verifier at the merge. The verifier posts one comment that names the PR and the merge commit and holds the criterion table, ticks the acceptance boxes it verified, and closes the issue. The issue stays open only while a row is `unmet`, or `not run` without another open issue that owns that work. Nobody collects box evidence or ticks boxes after the merge. (source: user, 2026-11-22)

## Worktrees and resources

59. Worktree location. Every worktree of this repository lives in `$WT`, the git-ignored `worktrees/` folder of the main checkout: issue, research, review, gate, verifier, binary and scratch worktrees alike, also the ones that live for minutes. One issue has one worktree, `$WT/issue-<n>-<slug>` (rule 1). Create issue and research worktrees with `wt-new.sh` and any other with `git -C <an existing worktree> worktree add [--detach] "$WT/<purpose>-<id>" <ref>`. Never give a path outside `$WT`: not the gates folder, not `/tmp`, not the work folder. Remove a worktree with `wt-done.sh` or `git worktree remove` as soon as its purpose ends, and before you hand back. `git worktree list` shows no path outside `$WT` except the main checkout. (source: user, 2026-11-22; a verifier's scratch merge worktree left in the gates folder, 2026-11-22)
60. Worktree base. A worktree is created from the tip of the integration branch (`git fetch`, then `origin/<base>` with `--no-track`), never from a stale local branch. One issue, one branch, one worktree; a second issue never reuses a worktree. (source: rule 1; `wt-new.sh`)
61. The main checkout is off limits for work. No `git` or `cargo` with the main checkout as cwd: use `git -C <worktree>` or a cwd inside the worktree. Run workshop tools from a worktree on the integration tip (yours, or `TOOLS_WT` of `rules-local.md` before you have one), never from the main checkout: it holds the copy on `main`, which can be older. The tools' own ref-only calls through `$ROOT` (fetch, worktree add and remove, branch delete) are allowed; they never touch the main checkout's files or index. After any `cd`, check it: `cd <dir> || exit 1`. (source: rule 2; user, 2026-11-22)
62. Clean-up of an issue worktree at merge, extending rule 7. Remove a worktree and its branches only after its PR is merged, the worktree is clean, and no process of the owner runs in it. One path per command, never a glob, never while the owner agent is alive. (source: rule 7; global orchestration rules)
63. Stale-worktree audit. `_tools/wt-audit.sh` lists every worktree with path, branch, age (newest file mtime), clean or dirty, owner (from `worktree.md`, else `none`) and PR state. It never removes. The handoff pipeline runs it and reports worktrees with owner `none`, dirty and old, or merged and still present. (source: rule 28; user, 2026-11-22)
64. Resources before work. Before a worktree is created or a build starts, `_tools/resources.sh` checks: free space on `BUILD_VOLUME` is at least `DISK_MIN_FREE` (each worktree builds its own large target, rule 6); no more than `BUILD_SLOTS` builds run; no other cargo process of mine runs. A failed check stops the step and reports the measured value. (source: user, 2026-11-22)
65. Builds and tests stay in their lane. Builds run only through `BUILD_SLOT`, tests through `TEST_LOCK`, both pinned to `PIN_CORES_BUILD` so that `CORES_RESERVED` stay free for pilot, campaign and desktop units. Never widen the pin and never bypass the scripts (rules 23 and 25). A test binary is never started directly. (source: rules 23 and 25; user, 2026-11-22)
66. Concurrency and GPU. At most 10 agents and 3 tickets at once (rule 21). Device work takes `TEST_LOCK` and runs sanitizer-first (rule 31). While a GPU campaign runs, at most `CAMPAIGN_BUILD_CAP` agents build. (source: rules 21 and 31; `models.md`; user, 2026-11-22)

## Verification

67. Box ticking: tick an acceptance box only for a criterion you checked, matched by its exact text; the diff of the body shows only those boxes changed. The verifier ticks at the merge the boxes it verified; nobody ticks after the merge. Never tick by a blanket edit of the body: a body also holds test-plan and other boxes. Rule 50 applies to every body upload. (source: verifier brief; user, 2026-11-09; user, 2026-11-22)
68. Crossing pull requests: a clean GitHub merge does not prove a semantic merge: two pull requests that pass alone can fail together. Before a merge the verifier runs the fast lint set (`lints.sh`) on the PR head merged with the current base tip. After the merge it runs the lint set on the new tip, and a `--no-run` build when another open or just-merged pull request touched the same files. A red result blocks the merge, or marks the report not clean. (source: #840 with #865; #980 with #765)
69. The verifier did not write the change and edits no code. It runs cargo only through `rc-run.sh` and only for the merged-tree checks of rules 56 and 68; it runs no GPU command. Its routine is `08_close/stages/01_verifier`. (source: user, 2026-11-01; user, 2026-11-22)
70. A test ran only when its line says so: a test passed only when the log holds its own `test <name> ... ok` line. A `test result: ok` line with `0 passed`, or with the test counted as ignored, means not run. A test marked `#[ignore]` because it needs an environment variable or a local file runs with that variable set and `--include-ignored`; the claim names the variable and the number of tests run. (source: final-gate audit, 2026-11-22: two ignored golden tests were reported as passed)
71. Evidence from the head that merges: a log is evidence only for the commit it ran on; `rc-run.sh` writes that commit into the log. When code paths changed between that commit and the PR head, the claim is stale: rerun and log it (rule 34). (source: `04_gate/stages/04_tested_vs_merged` applied to one pull request; derived, not a quoted rule)
72. Default behaviour: a change keeps the default behaviour bit-identical unless its issue says otherwise: archive hash, finalist, checkpoint bytes, config hash and JSON fields. Results stay deterministic per seed, a resumed checkpoint gives the uninterrupted result, and lockbox and leakage rules stay untouched. A test proves each property the change can affect. (source: strong-candidate brief, 2026-11-12)

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
| 2026-11-03 | tester bør ikke være ustabile. hvis de er ustabile så tester de ikke noenting | 57 |
| 2026-11-09 | bruk nettsøk så ofte du kan om det er nyttig, det er viktigste regelen | 19 |
| 2026-11-14 | Det er helt forferdelig fordi jeg ba om mest mulig helt fra starten. 2011 eller gjerne før også | 44 |
| 2026-11-14 | husk at sharpe skal måles mot noe halal, altså ikke national bonds som har riba. | 45 |
| 2026-11-22 | bruk sonnet agenter til utføring, opus til planlegging. Men la det du har startet kjøre, dette gjelder kun nye agenter. | 19 |
| 2026-11-22 | fable for planlegging av ekstraordinært vanskelig eller krevende saker, opus for planlegging forøvrig og sonnet til vanlig utføring. om man ikke får beskjed ellers. | 19 |
| 2026-11-22 | må du sende denne beskjeden hele tiden? kan du ikke vente til den er ferdig? | 52 |
| 2026-11-22 | på klasse A hvis de er merget så er de også allerede verifisert. | 58 |
| 2026-11-22 | hvis man har worktrees så bør det legges inn i en gitignored worktrees mappe | 59 |
