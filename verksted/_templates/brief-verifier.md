# Brief verifier section: verify and merge PR <PR> for issue <N>

Hard rules: never `pkill` or `killall` (kill only a pid you started, with `kill <pid>`); a wait uses `wait-for.sh`, never `pgrep -f`; no Python. A lint in `verksted_tools` fails on `pkill` and `killall` in scripts.

Used with `../_shared/brief.md`; `../_tools/brief.sh verifier --issue <N> --pr <PR>` prints the dispatch prompt. Replace every angle-bracket placeholder.

Follow `../08_close/CONTEXT.md` and its three stages exactly. Your scratch worktree is `$WT/verifier-<PR>`; `verify-merged.sh` creates it and `--done --out "$OUT"` removes it before your report (rule 59). Write `verification.md`, `report.md` and `check.md` to `OUT` (printed by the prompt), never into the worktree. Base branch `$BASE`, never `main`. You edit no code.

Claims file: one row per line `<label>|<log path>|<ERE>` for each claim of the hand-back (test counts, red runs, sanitizer 0 errors, warnings 0, speedups, wall times), written before step 3 of `stages/01_verifier`.

Final report, only after `verifier-done.sh "$N" "$PR" "$OUT"` exits 0: merged sha (or the hold and why), the claims you verified, the claims left unverified, whether the checks on the new tip are clean, and anything unexpected.
