# 01_verifier: check the evidence on the tree that merges, merge, tick, comment

One job: verify a pull request on the tree that merges, merge it, tick the verified boxes and post one comment.

A run ends in one of three outcomes: merged, hold, or conflict (also a hold). A hold keeps the PR open and posts the same three records.

## Inputs

**This run:**
- `PR`, `N`, `EPIC` and `OUT` from the dispatch prompt (`brief.sh verifier`); `OUT` is the durable record directory under the local area
- the claims file of the hand-back: one row per line `<label>|<log path>|<ERE>`
- `gh issue view "$N" --json body,comments`

**Reference (every run):**
- `../../../_shared/rules.md` (rules 5, 13, 34, 35, 41, 56, 58, 59, 61, 67 to 71)
- `../../../_templates/verification-comment.md`
- `../../../_templates/merge-body.md`
- `../../../_tools/verifier-check.sh`
- `../../../_tools/verify-merged.sh`
- `../../../_tools/verifier-done.sh`
- `../../../_tools/test-summary.sh`
- `../../CONTEXT.md`

Do NOT load: editing code, GPU commands, the implementer's prose beyond the claims, the author worktree after the merge.

## Process

1. If an input is missing or empty, stop and say which.
2. `mkdir -p "$OUT"`. The three records live in `$OUT`, never in `$WT/verifier-$PR`: `verify-merged.sh --done` removes that worktree. A rerun overwrites the files, never appends. Run every command from `$WT/verifier-$PR` or with absolute paths, never from the main checkout (rule 61). After any `cd`, check it: `cd <dir> || exit 1`. The verifier did not write the change and edits no code (rule 69).
3. `verksted/_tools/verifier-check.sh "$PR" --claims <file> --scope <glob>...`: state OPEN, base `$BASE`, every changed file in scope, no reverted base file, the author worktree clean at the PR head (the merge removes it, F60), every claim `found`. On CONFLICTING: finish steps 4 and 5, then stop without merging: the outcome is a hold with the reason `CONFLICTING`, the files and the evidence verdict; the author merges the base. Continue at step 10.
4. Evidence: re-read every cited log with `verksted/_tools/test-summary.sh <log> [--expect <test name>]...`. A test claim matches the test's own `test <name> ... ok` line, never a `test result:` line (rule 70). A claim with no log is unverified; `empty` means no test ran; `stale` means the log ran on older code than the PR head (rule 71); a log with `unknown head` predates the `RC_HEAD` line, and a rerun is asked for only when code paths changed after the pull request was opened. A contradicted, empty or stale claim is a hold. A diff that adds a test file needs a `sanitizer_plan` claim with `0 failed`.
5. Criteria: copy `verification-comment.md` to `$OUT/verification.md`; one row per acceptance criterion: `met` with its evidence, `not run (#M)` with the reason and the open issue that owns the work (a criterion that needs the full gate, a GPU run or a measurement is such a row), or `unmet`. A criterion with no checkable evidence stays unticked and is a `not run (#M)` or `unmet` row.
6. Merged-tree checks: `verksted/_tools/verify-merged.sh "$PR" --lints`, plus `--goldens` when the diff changes a hashed byte format (rule 56) and `--walk` when it changes `verksted/`. `--walk` runs the strict walk: every failure is a hold. The tool creates `$WT/verifier-$PR` at the base tip and merges the PR head without a commit. The job can run longer than 10 minutes under contention: start the tool once in a background shell, then use one long tracked wait on its completion marker, `verksted/_tools/wait-for.sh --log "$G/verifier$PR-merged.log" --pattern '^RC_DONE rc='` under a timeout of at least 3600 s. Never a short timeout, never a second run: a second run hits the duplicate-job lock (exit 3, rule 53). The background shell's exit status is the result. Exit 75 means the slot gave up: run it once more (rule 53). Red or a conflict: do not merge; the outcome is a hold with the failing tests or files.
7. Merge, only when nothing above held: `gh pr merge "$PR" --squash --delete-branch --subject "<type(area): subject> (#$PR)" --body-file <file>` with the body from `merge-body.md`, into `$BASE` only (rule 41). The merge removes the author worktree: never `cd` into it afterwards.
8. After the merge: `verksted/_tools/verify-merged.sh "$PR" --tip --lints`, plus `--build` when another open or just-merged pull request touched the same files (rule 68). Red: the report is not clean; report it, do not fix it.
9. Tick the acceptance box of each `met` row in the issue body, matched by its exact text, never a `not run` or `unmet` row (rule 67). Download the body, change only those boxes, check `test -s F && grep -q '^## ' F`, upload with `gh issue edit "$N" --body-file F` (rule 50). Fill line 1 of `$OUT/verification.md` (`Merged into <base> as <short sha> (PR #<PR>).`) and the verdict; post it once with `gh issue comment "$N" --body-file`. Only verified facts or attributed claims (rule 35). Nobody ticks after this step. A hold ticks nothing and uses the hold line and `Verdict: HOLD`; it still posts the comment.
10. Write `$OUT/report.md`: merged sha or the reason for the hold, the claims verified, the claims unverified, whether the tip checks are clean, findings outside the pull request. A hold after any earlier step still writes `verification.md` and `report.md`.
11. Run `verksted/_tools/verifier-done.sh "$N" "$PR" "$OUT"`, then `verksted/_tools/verify-merged.sh "$PR" --done --out "$OUT"`. Report done only when `verifier-done.sh` exits 0; on exit 1 fix the named file or comment and run it again, never rewrite `check.md`.

## Outputs

- `$OUT/verification.md`
- `$OUT/report.md`
- `$OUT/check.md`

`$OUT` is `$VLOCAL/verifier/$PR`. `verify-merged.sh --done --out "$OUT"` refuses to remove the worktree while one of the three files is missing or empty.

## Check

```sh
verksted/_tools/verifier-done.sh "$N" "$PR" "$OUT"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `$OUT/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green. The script passes the merged form (PR MERGED, `Merged into ...`, `Verdict: CLOSE` or `KEEP OPEN`, met rows equal ticked boxes) and the hold form (PR OPEN, `Hold: PR #<PR> kept open, ...`, `Verdict: HOLD`), and it reads no current branch.
