# 02_brief: one complete brief per ticket

One job: write the brief of each ticket from the template.

## Inputs

**This run:**
- `../01_wave_plan/output/wave.md`
- `../01_wave_plan/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_templates/brief.md`, `../../../_templates/brief-research.md`, `../../../_templates/brief-clerk.md`
- `../../../_tools/brief.sh`
- `../../../_shared/brief.md`
- `../../CONTEXT.md`

Do NOT load: other tickets' briefs, the agents' reports.

## Process

1. If an input is missing or empty, stop and say which.
2. For each row of `wave.md` run `verksted/_tools/brief.sh impl|research|clerk --issue <N> ...` and save the prompt as `output/brief-<N>.md`. Fill the ticket section of the template (`_templates/brief.md`, `brief-research.md` or `brief-clerk.md`) with the owned and forbidden files, the design link and the acceptance criteria, and append only those field lines (the steps stay in the template, which the agent reads). A research brief says "no code, no branch, no commit". An implementation brief quotes the decision comment URL.
3. Line 1 of every brief is the no-resume line "You are not resumed automatically; block in your own turn until everything is done." The prompt names the values N, SLUG, W, BRANCH, BASE, G, EPIC and PR. `brief.sh` refuses a missing value and a leftover placeholder; never pass a literal `<N>`.

## Outputs

- `brief-<N>.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
for f in "$V"/03_orchestrate/stages/02_brief/output/brief-*.md; do head -1 "$f" | grep -qxF 'You are not resumed automatically; block in your own turn until everything is done.' || exit 1; grep -q "^Read and follow verksted/_shared/brief.md" "$f" || exit 1; if grep -q "^Kind: impl" "$f"; then for k in 'Worktree:' 'Owned files:' 'Forbidden files:'; do grep -q "^$k" "$f" || exit 1; done; fi; ! grep -q '<' "$f" || exit 1; done; ls "$V"/03_orchestrate/stages/02_brief/output/brief-*.md > /dev/null
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
