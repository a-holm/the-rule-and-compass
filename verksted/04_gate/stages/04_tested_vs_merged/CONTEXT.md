# 04_tested_vs_merged: the tested tree is the merged tree

One job: show that nothing but docs changed after the gate.

## Inputs

**This run:**
- `../03_full_gate/output/gate.md`
- `../03_full_gate/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: new features, a second gate.

## Process

1. If an input is missing or empty, stop and say which.
2. Read `tested: <sha>` from `gate.md`. Run `git diff --stat <sha> "$BASE" -- my-version scripts` and save it as `diff.md`. The diff of code paths must be empty.
3. If code changed after the gate, the gate result is void for that change: return to stage 03.

## Outputs

- `diff.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
git diff --quiet "$(sed -n 's/^tested: //p' "$V/04_gate/stages/03_full_gate/output/gate.md")" "$BASE" -- my-version scripts
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
