# Brief clerk section: merge PR <PR> for issue <N>

Used with `../_shared/brief.md`; `../_tools/brief.sh clerk --issue <N> --pr <PR>` prints the dispatch prompt. Replace every angle-bracket placeholder.

Repo root, GitHub repo, use `gh`. Base branch `$BASE` (`integration/gpu-hardening`). Never merge to `main`. Never force-push, never rebase, never `git stash`, no Python, no pkill. Do not run cargo, do not edit code; you only verify and merge. The orchestrator already accepted the PR's design deviations; your job is to check the evidence and merge.

For the PR you are given:
1. Write the claims of the hand-back summary to a claims file, one row per line `<label>|<log path>|<ERE>` (test counts, red runs, sanitizer 0 errors, warnings 0, speedups, wall times). Run `verksted/_tools/clerk-check.sh <PR> --claims <file>`. It fetches origin and prints state, base, mergeable, head, the changed files, the diff stat, the worktree of the head branch (must be clean and at the PR head: the merge removes it), the conflicts and one row per claim.
2. State must be OPEN and the base `$BASE`; the tool retries while mergeable is UNKNOWN. If it says CONFLICTING: still check scope and evidence, then stop without merging and report "CONFLICTING", the conflicting files and your evidence verdict, so the PR agent can merge the base itself.
3. Own changes must match the issue's scope: no unrelated files (`out of scope` rows), no reverted base files. Report anything unexpected and do not merge.
4. Evidence: every claim must be `found` in a cited log under `$GATES`. Re-read counts with `verksted/_tools/test-summary.sh <log>`. A claim with no log is "unverified". A contradicted claim blocks the merge: report it.
5. Merge: `gh pr merge <PR> --squash --delete-branch --subject "<type(area): subject> (#<PR>)" --body-file <file>`, into `$BASE` only, never `main`. The body is plain technical English, no em-dashes, no AI attribution: what changed (2 to 4 short paragraphs), the evidence you verified (counts with log file names), unverified numbers attributed ("the PR reports ..."), last line `Refs #<N>, #<epic>`. Never write predictions or plans as facts.
6. After the merge: `git fetch origin`; the new sha is `git rev-parse --short origin/$BASE`. Comment on the issue: `gh issue comment <N> --body-file <file>` with "Merged into $BASE as <sha> (PR #<PR>)." plus the verified facts and the open items as present state ("not run", "open", "tracked by #N"). Only verified facts or attributed claims.
7. Clean up with `verksted/_tools/wt-done.sh <N> --pr <PR>`. It removes the worktree and branches only when the worktree is clean and the PR is merged. `gh pr merge --delete-branch` usually removed the worktree already: that is expected, not an error. Do not touch any other worktree.

Final report: merged sha (or why not merged), the claims you verified, the claims left unverified, and anything unexpected.
