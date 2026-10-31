# 02_integration_merge: every ticket is in the base

One job: confirm that all gate tickets are merged into the integration branch.

## Inputs

**This run:**
- `../01_targeted_tests/output/tickets.md`
- `../01_targeted_tests/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: gate runs, unmerged branches.

## Process

1. If an input is missing or empty, stop and say which.
2. Merging is git work in the integration worktree, ticket by ticket, without a gate per merge. When `main` moved, merge `origin/main` into `BASE` instead of rebasing.
3. New tickets branch from the tip of `BASE`, not from `main`.
4. Write `merged.md`: the output of `git log --oneline main..BASE`.

## Outputs

- `merged.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
for n in $(grep -o '^| #[0-9]*' "$V/04_gate/stages/01_targeted_tests/output/tickets.md" | tr -d '| #'); do git log --format=%B main.."$BASE" | grep -q "Refs #$n" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
