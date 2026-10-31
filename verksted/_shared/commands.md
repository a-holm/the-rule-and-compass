# Commands: exact and copyable

Commands that work on this machine. Run them from the run root (`RUN`) unless a line says otherwise. Longer explanations live in `../../docs/issue-workflow.md`. Hard rules and their sources: `rules.md`.

## Run variables

```sh
ROOT=$(git worktree list --porcelain | sed -n "1s/^worktree //p")   # main checkout: read only
WT=$ROOT/worktrees
N=368 SLUG=verksted                         # issue number and kebab-case slug
BASE=integration/gpu-hardening              # or main
W=$WT/issue-$N-$SLUG                        # this ticket's worktree
RUN=$W                                      # checkout that holds this run's output/ files
V=$RUN/verksted
L=/mnt/lager/tmp/opencode/one-full-suite.sh
G=/mnt/lager/tmp/eureqa-gates               # logs: $G/issue$N-<what>.log
PAUSE=/mnt/lager/tmp/opencode/PAUSE
```

Data and run variables: `DATA`, `STORE`, `ENRICHED`, `MEMBERSHIP`, `TARGET`, `LABEL`. Memory file: `MEM`. Their values come from `../setup/questionnaire.md` or the dispatcher.

## Check a stage

```sh
verksted/_tools/check.sh "$V/<pipeline>/stages/<stage>/output" '<the stage Check command>'
```

It writes `check.md` (line 1 `pass` or `fail: exit <code>: <last output line>`) and exits with the command's code. Export the run variables first.

## Worktree

```sh
git -C "$ROOT" fetch origin
git -C "$ROOT" worktree add -b "issue/$N-$SLUG" "$W" "$BASE"      # branch from the local tip of BASE
git -C "$ROOT" worktree add --detach "$WT/research-$N" "$BASE"    # research run: no branch
git -C "$ROOT" worktree list
```

New tickets branch from the integration branch, not from `main`. Do not set `CARGO_TARGET_DIR`: each worktree keeps its own `my-version/target/`.

## Build and test (always through the lock)

```sh
$L bash -c "cd $W/my-version && CARGO_BUILD_JOBS=2 timeout 580 cargo test --release --test <file> <filter>" > "$G/issue$N-<what>.log" 2>&1
$L bash -c "cd $W/my-version && CARGO_BUILD_JOBS=2 RCSR_CUDA_ARCH=sm_86 timeout 580 cargo test --release --features cuda --test <file> <filter>" > "$G/issue$N-<what>.log" 2>&1
ps -eo pcpu,args | grep -E 'cargo|rustc'          # look first: one cargo at a time
```

The lock runs the command at nice 19 and ionice idle, waits while `$PAUSE` exists, and gives up after 7200 s with exit 75 (rerun then). Keep each hold under 10 minutes. Count results with `grep -c 'test result: ok' <log>` and `grep -c 'test result: FAILED' <log>`.

## Full gate (once, on the integration tip)

```sh
verksted/_tools/gate.sh "$WT/integration-gpu-hardening" <tag>   # logs: $G/<tag>-{cpu,cuda,bench}.log
```

Three lock holds: `cargo test --release`; `RCSR_CUDA_ARCH=sm_86 cargo test --release --features cuda`; `cargo build --release && ./target/release/rcsr-bench --tiny` (must print `recovered 27/27`). Start it as one background shell and wait for the notification.

## Waiting

One blocking call, or one background shell and its notification. Wait on a pid or a file, never on a process name.

```sh
tail --pid="$PID" -f /dev/null          # blocks until the pid you started exits
kill -0 "$PID"                          # alive? exit 0 yes
verksted/_tools/watch.sh -m 45 "$WT"/issue-*   # silence watcher: one background shell, prints SILENT <n>min: <path>
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
gh pr create --base "$BASE" --title "$(git log -1 --pretty=%s)" --body-file <filled pr.md>
gh pr view "issue/$N-$SLUG" --json state,baseRefName,headRefOid,body
```

## Merge

```sh
gh pr merge <pr> --squash --subject "fix(area): subject (#<pr>)" --body "<what changed, prove red, test counts>. Refs #$N"
gh pr merge <pr> --merge --subject "merge integration/<name>: <summary>" --body "<gate numbers>. Refs #<n1>, #<n2>"
gh pr view <pr> --json state,mergeCommit
```

Ticket PRs into the integration branch are squashed. The integration branch into `main` keeps its commits (`--merge`), as the history of `main` shows. Hand-write subject and body. Never `Closes`. No `--delete-branch` while the worktree exists (it fails on the checked-out branch).

## Close and clean up (this ticket only)

```sh
gh issue close "$N" --reason completed
git worktree remove "$W"
git branch -D "issue/$N-$SLUG"                  # a squash merge makes -d refuse
git push origin --delete "issue/$N-$SLUG"
git worktree prune
test ! -e "$W"
```

## Rust experiments

```sh
nice -n 19 rustc -O /mnt/lager/tmp/<scratch>/exp.rs -o /mnt/lager/tmp/<scratch>/exp   # single file, outside the repo
perl -0pi -e 's/old/new/' <file>                                                      # scripted edits: perl or sed, never python
git ls-files '*.py'                                                                   # must print nothing
```

## Data and runs

```sh
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

`rcsr` and `rcsr-launch` are in `my-version/target/release/` of the worktree.

## Machine

```sh
touch "$PAUSE"; rm "$PAUSE"                     # pause new runs, resume them
nproc; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -8
```

## Structure check of this workshop

```sh
verksted/_tools/links.sh
```
