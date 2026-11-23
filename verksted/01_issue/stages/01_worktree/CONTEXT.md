# 01_worktree: own checkout for one issue

One job: create the worktree and branch for the issue and record them.

## Inputs

**This run:**
- `gh issue view "$N" --json title,state` (the issue must be open)
- `../../../_shared/local.md`: the local area and the `machine.md` keys that `resources.sh` reads

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`
- `../../../_tools/wt-new.sh`
- `../../../_tools/resources.sh`

Do NOT load: the issue discussion, other worktrees, the code.

## Process

1. If `N`, `SLUG` or `BASE` is missing, or the issue is not open, stop and say which.
2. Run `verksted/_tools/resources.sh`. Stop on a red check and report the measured value (rule 64).
3. Run `verksted/_tools/wt-new.sh "$N" "$SLUG" --base "$BASE"`. It fetches, branches from `origin/$BASE` with `--no-track`, adds the worktree and creates the scratch dir `$GATES/issue$N-scratch/`. The path must be `worktrees/issue-$N-$SLUG`. Never commit on `main`.
4. Create `output/` inside the new worktree and write `worktree.md`: path, branch, base tip sha and the line `owner: <agent id or role>` (`wt-audit.sh` reads it).
5. Own `<crate>/target/` (`verksted/_tools/crate-dir.sh "$W"`): never set `CARGO_TARGET_DIR`.

References: [rules 59 to 66](../../../_shared/rules.md) (location, base, main checkout, clean-up, audit, resources, lanes, concurrency).

## Outputs

- `worktree.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
case "$W" in "$ROOT"/worktrees/issue-"$N"-*) ;; *) exit 1 ;; esac; test "$(git -C "$W" branch --show-current)" = "issue/$N-$SLUG" && git -C "$ROOT" worktree list --porcelain | grep -qxF "worktree $W" && git -C "$W" merge-base --is-ancestor "$BASE" HEAD
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
