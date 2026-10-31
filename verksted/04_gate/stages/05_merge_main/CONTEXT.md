# 05_merge_main: land the integration branch

One job: merge the integration pull request into main and report.

## Inputs

**This run:**
- `../04_tested_vs_merged/output/diff.md`
- `../04_tested_vs_merged/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: re-gating, new tickets.

## Process

1. If an input is missing or empty, stop and say which.
2. Open the PR `BASE` -> `main` with the gate numbers and one `Refs #N` line per ticket. Merge it alone, never at the same time as another merge. `Closes` is forbidden.
3. Use `gh pr merge <pr> --merge` with a hand-written subject and body (section "Merge" of `commands.md`). Then `git fetch` and check the merge sha is on `origin/main`.
4. Write `report.md`: gate numbers, tested sha, merge sha. Closing the issues is `../../08_close/CONTEXT.md`.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
pr=$(gh pr list --base main --head "$BASE" --state merged --json number -q '.[0].number'); sha=$(gh pr view "$pr" --json mergeCommit -q .mergeCommit.oid); git fetch origin main && git merge-base --is-ancestor "$sha" origin/main
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
