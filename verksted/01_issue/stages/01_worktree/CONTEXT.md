# 01_worktree: own checkout for one issue

One job: create the worktree and branch for the issue and record them.

## Inputs

**This run:**
- `gh issue view "$N" --json title,state` (the issue must be open)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: the issue discussion, other worktrees, the code.

## Process

1. If `N`, `SLUG` or `BASE` is missing, or the issue is not open, stop and say which.
2. Create the worktree and branch from the local tip of `BASE` (section "Worktree" of `commands.md`). Never commit on `main`.
3. Create `output/` inside the new worktree and write `worktree.md`: path, branch, base tip sha.
4. Own `my-version/target/`: never set `CARGO_TARGET_DIR`.

## Outputs

- `worktree.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test "$(git -C "$W" branch --show-current)" = "issue/$N-$SLUG" && git -C "$ROOT" worktree list --porcelain | grep -qxF "worktree $W" && git -C "$W" merge-base --is-ancestor "$BASE" HEAD
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
