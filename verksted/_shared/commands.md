---
type: reference
description: Exact, copyable commands for the run variables, builds, tests, gates, waiting, git, data, the machine and the structure check.
---

# Commands: exact and copyable

Commands that work on this machine. Run them from the run root (`RUN`) unless a line says otherwise. Longer explanations live in `../../docs/issue-workflow.md`. Hard rules and their sources: `rules.md`. Every script prints its usage with `-h`; `../_index.md` lists them all.

## Run variables

```sh
source verksted/_tools/lib.sh               # defines ROOT, WT, BASE, GATES, G, B, L, PAUSE, SANITIZER, CUDA_ARCH
N=<issue> SLUG=<slug>                       # issue number and kebab-case slug
W=$WT/issue-$N-$SLUG                        # this ticket's worktree
RUN=$W                                      # checkout that holds this run's output/ files
V=$RUN/verksted
```

`<crate>` is the engine crate directory of a worktree: the output of `verksted/_tools/crate-dir.sh "$W"` (the one first-level directory whose `Cargo.toml` declares package `rule-and-compass`; `RCSR_CRATE` names it instead). No tracked text names the directory. Print it, or enter it:

```sh
verksted/_tools/crate-dir.sh "$W"
cd "$W/$(verksted/_tools/crate-dir.sh "$W")"
```

`GATES`, `MACHINE_SCRIPTS`, `SANITIZER` and `CUDA_ARCH` come from `machine.md` (`_shared/local.md`); a missing key stops the script with exit 2. Logs go to `$G/issue$N-<what>.log`. `B` is the build slot, `L` the test lock; use them through `rc-run.sh`. Every `RCSR_*` override is listed in `lib.sh`.

Data and run variables: `DATA`, `STORE`, `ENRICHED`, `MEMBERSHIP`, `TARGET`, `LABEL`. Memory file: `MEM`. Their values come from `../setup/questionnaire.md` or the dispatcher.

## Check a stage

```sh
export N SLUG W V PR                       # every run variable the stage Check reads
verksted/_tools/check.sh --stage <pipeline>/<stage>
```

It reads the Check block of `$V/<pipeline>/stages/<stage>/CONTEXT.md`, sources `lib.sh`, exports the variables the block reads and runs it under `bash -u`. A variable still unset (`W`, `N`, `V`, ...) stops it with exit 2 and the name on stderr; a contract without exactly one Check block exits 2. It writes `$V/<pipeline>/stages/<stage>/output/check.md` (line 1 `pass` or `fail: exit <code>: <last output line>`) and exits with the block's code. Checks compare against `origin/$BASE` and fetch first; `verksted-check walk` fails a Check that names `$BASE` as a git ref without `origin/`. For a Check that lives outside a contract: `verksted/_tools/check.sh <output-dir> '<command>'`.

## Worktree

```sh
verksted/_tools/wt-new.sh "$N" "$SLUG" [--base <branch>]   # worktree, branch issue/N-SLUG, scratch dir; prints W= BRANCH= BASE_SHA= SCRATCH=
verksted/_tools/wt-new.sh --research "$SLUG"               # research run: detached, no branch
git worktree list
```

New tickets branch from the integration branch, not from `main`. Do not set `CARGO_TARGET_DIR`: each worktree keeps its own `<crate>/target/`.

## Build, test, run

```sh
verksted/_tools/rc-run.sh build "$W" "$N" build -- cargo test --release --no-run --test <file>          # build slot
verksted/_tools/rc-run.sh test "$W" "$N" <what> -- cargo test --release --test <file> <filter>          # CPU test: build slot
verksted/_tools/rc-run.sh test "$W" "$N" <what> --cuda -- cargo test --release --features cuda --test <file> <filter>   # GPU test: test lock
verksted/_tools/rc-run.sh run "$W" "$N" <what> -- ./target/release/rcsr <args>                          # rcsr and rcsr-bench: test lock
# rc-run.sh lock modes run the command through verksted/_tools/lock-run.sh: its own session; after it ends, every process that shares the flock of that hold is TERMed then KILLed by pid (nothing is signalled unless the process itself owns the flock on the lock file)
# A monitor or background job started under a hold must use setsid and 9>&- (close the lock fd), or the reap ends it with the hold
# runlib.sh rl_gpu (and rl_chunks) holds run through lock-run.sh the same way; a runbook that takes the lock directly runs: bash verksted/_tools/lock-run.sh "$LOCKFILE" <command...> as the first command inside the lock script
ps -eo pcpu,args | grep -E 'cargo|rustc'          # look first: one cargo at a time
```

The log is `$G/issue$N-<tag>.log`; its last line is `RC_DONE rc=<n>`. Lock holds stay under 10 minutes. Exit 75 means the slot or lock gave up: rerun. Exit 70 means a new `NVRM: Xid` line appeared since the hold started: stop all GPU work and report. A falling total is not a fault (rule 31). Count results:

```sh
verksted/_tools/test-summary.sh "$G/issue$N-<what>.log"        # passed, failed, ignored, warnings
```

## Fast lint set (once per PR, on your head merged with the base tip)

```sh
verksted/_tools/lints.sh --tree "$W"        # prints the 12 `--test` arguments of the crate; one home of the list
verksted/_tools/rc-run.sh test "$W" "$N" lints -- cargo test --release $(verksted/_tools/lints.sh --tree "$W")
verksted/_tools/rc-run.sh cpu "$W" "$N" verksted -- bash -c 'cd "$1/verksted/_tools/verksted-check" && cargo test --release' _ "$W"
verksted/_tools/links.sh --strict
```

The 12 crate targets plus `cargo test --release` in `verksted/_tools/verksted-check/` (links, walk and tools guard tests) are the whole set. A new test file that spawns processes needs a line in `<crate>/tests/sanitizer.plan`.

While the desktop streams video the scripts pin work to `PIN_CORES_BUSY`, the build slots drop to `BUILD_SLOTS_BUSY` and `RUST_TEST_THREADS=4`. Force the streaming restriction, clear it, and check it:

```sh
touch "$STREAMING"
rm "$STREAMING"
"$OC/desktop-busy.sh"; echo $?   # 0 = desktop busy, 1 = free
```

## Full gate (once, on the integration tip)

```sh
verksted/_tools/gate.sh "$WT/integration-gpu-hardening" <tag>   # logs: $G/<tag>-{cpu,cuda,bench}.log
```

Three lock holds: `cargo test --release --no-fail-fast`; `RCSR_CUDA_ARCH=$CUDA_ARCH cargo test --release --features cuda --no-fail-fast`; `cargo build --release && ./target/release/rcsr-bench --tiny` (must print `recovered 27/27`). Start it as one background shell and wait for the notification.

## Waiting

One blocking call, or one background shell and its notification. A wait always uses `wait-for.sh`: on a log line, a file, a count, an exact pid or an executable path. Never write `pgrep -f` (it matches the waiting shell itself; the lint `tools_use_no_pgrep_f` fails on it).

```sh
verksted/_tools/wait-for.sh --log "$G/issue$N-<what>.log"       # until RC_DONE; prints the test summary
verksted/_tools/wait-for.sh --file <path> --timeout 3000
verksted/_tools/wait-for.sh --count <n> --dir <dir> --glob '*.done'
verksted/_tools/wait-for.sh --pid "$PID" --timeout 3000       # until that exact pid has ended (reads /proc, no command line)
verksted/_tools/wait-for.sh --exe /abs/path/to/binary          # until no process runs that executable (exact /proc/<pid>/exe)
tail --pid="$PID" -f /dev/null          # blocks until the pid you started exits
kill -0 "$PID"                          # alive? exit 0 yes
verksted/_tools/watch.sh -m 45 "$WT"/issue-*   # silence watcher: one background shell, prints SILENT <n>min: <path>
verksted/_tools/status.sh                       # one screen: origin tip, open PRs, agents, machine
```

## Debug a hang

```sh
RCSR_ATTACHABLE=1 ./target/release/rcsr ... &                   # opt in to being traced (ptrace_scope=1)
scripts/stacks.sh <pid> [out-file]                              # exit 0 gdb, 3 eu-stack, 4 /proc table only
$L scripts/stall-watch.sh -i 60 -m 10 -l <log> -- bash -c "cargo test --release ..."
timeout -s KILL 300 <test-bin> <test> --exact --nocapture --test-threads=1
gdb -q -batch -nx -ex run -ex 'thread apply all bt 25' --args <test-bin> <test> --exact --nocapture --test-threads=1
```

## Commits, push, pull request

```sh
git commit -m "fix(gpu-enrich): short imperative subject" -m "Refs #$N"
git push -u origin "issue/$N-$SLUG"
git merge "origin/$BASE"                                        # when the base moved; never rebase after a push
verksted/_tools/leak-scan.sh --diff "origin/$BASE..HEAD"        # before the PR: added lines and commit messages
gh pr create --base "$BASE" --title "$(git log -1 --pretty=%s)" --body-file <filled pr.md>
gh pr view "issue/$N-$SLUG" --json state,baseRefName,headRefOid,body
```

## Leak scan

```sh
verksted/_tools/leak-scan.sh                                    # every tracked file, path and text
verksted/_tools/leak-scan.sh --diff "origin/$BASE..HEAD"        # a PR: added lines and commit messages
```

The word list lives outside git (`$RCSR_BANNED_WORDS`, else `~/.config/rcsr/banned-words`). Exit 0 clean, 1 hits, 3 no list.

## Merge

```sh
verksted/_tools/verifier-check.sh <pr> --claims <file>             # read-only evidence check before the merge
gh pr merge <pr> --squash --delete-branch --subject "fix(area): subject (#<pr>)" --body-file <file>
gh pr view <pr> --json state,mergeCommit
```

Ticket PRs merge into the integration branch only, squashed, by the verifier. `main` only on the lead's written instruction. Hand-write subject and body. Never `Closes`. The worktree is removed afterwards with `wt-done.sh` (next section).

`gh pr merge --delete-branch` also removes the worktree that has the head branch checked out (gh 2.103.3; the exact conditions are not verified), so `verifier-check.sh` checks before the merge that this worktree is clean and at the PR head. Afterwards an already-gone worktree is expected, not an error.

## Close and clean up (this ticket only)

```sh
gh issue close "$N" --reason completed
verksted/_tools/wt-done.sh "$N" --dry-run       # prints the actions
verksted/_tools/wt-done.sh "$N"                 # worktree, local and remote branch; refuses unless clean and merged
verksted/_tools/wt-done.sh "$N" --pr <pr>       # worktree already removed by the merge: deletes leftover branches, exit 0 "already gone" when the PR is MERGED
test ! -e "$W"
```

## Rust experiments

```sh
nice -n 19 rustc -O "$GATES/issue$N-scratch/exp.rs" -o "$GATES/issue$N-scratch/exp"   # single file, outside the repo
perl -0pi -e 's/old/new/' <file>                                                      # scripted edits: perl or sed, never python
git ls-files '*.py'                                                                   # must print nothing
```

## GPU safety

The build selects the newest CUDA 13 toolkit by rule (`docs/gpu-backend.md`); the sanitizer is `$SANITIZER` (`machine.md`); the system 12.0 `compute-sanitizer` cannot attach on this driver.

```sh
verksted/_tools/sanitize.sh "$W" "$N" memcheck --test <file> -- <filter>    # builds, then runs under compute-sanitizer through the lock; exit 0 before any plain GPU run
# search-level tests: the config helper must set dispatch: DispatchMode::Device (CLI: --dispatch device), else the sanitized run stays on the CPU
journalctl -k --since "<start>" | grep -i xid   # must print nothing
journalctl -k --since "<start>" | grep -c 'NVRM: Xid'   # new lines since the start time of the step, never a journal total
verksted/_tools/sanitize.sh "$W" "$N" memcheck --test <file> --api-errors no -- <test> --exact   # one test with driver API error reporting off
verksted/_tools/sanitize-all.sh "$W" "$N" [--only <binary>]... [--dry-run]   # the planned memcheck pass (<crate>/tests/sanitizer.plan), one lock hold per run, summary log <id>-san-summary; a run-bin line runs a release binary with stated arguments; gate.sh has no sanitizer stage
```

## Pinned binaries and paired runs

```sh
verksted/_tools/bin-pin.sh "$W" <dest-dir> rcsr                 # copy, sha256, revision; runs it with --version
verksted/_tools/ab-pair.sh --label <name> --a <bin> --b <bin> --seeds 1,2,3 -- <args with {seed}>
```

## Data and runs

```sh
verksted/_tools/local-init.sh --dataset "$DS" --card <card.md>        # creates the local area and datasets/<id>/; never overwrites
verksted/_tools/data-profile.sh --dir "$DATA" --dev-end <date> [--tz <offset>] --out <profile.md>   # stage 03: names, header, counts; no value row
verksted/_tools/ohlcv-view.sh --variant ohlcv --series single              # stage 04_ohlcv: feature_groups and view_excludes
rcsr lock check --record <record> <store dir>                              # stage 07: exit 0 clean, 4 a finding
verksted/_tools/run-record.sh new --id <YYYYMMDD-slug> --kind store-build --bin <path> --worktree "$W"
verksted/_tools/run-record.sh close --id <id> --log <log>
verksted/_tools/run-record.sh latest [--kind <kind>]                      # the last run card
rcsr import in.csv "$STORE" --group-rows 8192
rcsr prep --input "$DATA" --out "$STORE" --resample 1h
rcsr panel membership --data daily.csv --top 100 --lookback 90 --min-history 90 --rebalance 30 --out "$MEMBERSHIP"
rcsr enrich --data "$STORE" --out "$ENRICHED" --features all --gpu on
rcsr audit --data "$ENRICHED" --target "$TARGET" --time ts --horizon <n> --json
rcsr verify-manifest "$ENRICHED.manifest.json" --data "$ENRICHED"
rcsr-launch --dry-run --data "$ENRICHED" --target "$TARGET" --vars <a,b> --backend gpu --window <label,start,end> --out-dir runs
rcsr dev-report "runs/$LABEL.dev.ndjson"
rcsr schema
```

`rcsr` and `rcsr-launch` are in `"$(verksted/_tools/crate-dir.sh "$W")/target/release/"` of the worktree.

## Dispatch an agent

```sh
verksted/_tools/brief.sh impl --issue "$N" --slug "$SLUG" --epic <n>     # prints the dispatch prompt
verksted/_tools/brief.sh research --issue "$N" --slug "$SLUG"
verksted/_tools/brief.sh verifier --issue "$N" --pr <pr>
verksted/_tools/issue-lint.sh <body.md> --number "$N"                    # issue body against the template
```

## Machine

```sh
touch "$PAUSE"; rm "$PAUSE"                     # pause new runs, resume them
verksted/_tools/machine-sync.sh --check         # live machine scripts against verksted/_tools/machine/
nproc; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -8
```

## Structure check of this workshop

```sh
verksted/_tools/links.sh --strict
verksted/_tools/rc-run.sh cpu "$W" "$N" external -- bash -c 'cd "$1/verksted/_tools/verksted-check" && cargo run --release -- links --external "$1/verksted"' _ "$W"
```

`links --external` also checks every path outside `verksted/` strictly and resolves `<crate>` through `crate-dir.sh`; `gate.sh` runs it.

`lib.sh` and `runlib.sh` are sourced by the other scripts and by long runs (`../07_longrun/CONTEXT.md`).
