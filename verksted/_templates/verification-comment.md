Merged into <base> as <short sha> (PR #<PR>).

# Verification: <issue number and title>

Instructions, delete this paragraph and the status paragraph before posting: copy to `$OUT/verification.md` (`OUT` is printed by `brief.sh verifier`), fill in, and post it as one issue comment. The verifier did not implement the issue and edits no code. For a hold, replace line 1 by `Hold: PR #<PR> kept open, <reasons>.` and end with `Verdict: HOLD` (the PR stays OPEN, no box is ticked).

| Row | Criterion | Evidence | Status |
|---|---|---|---|
| AC1 | <criterion text> | <test name, file:line or log path> | met |
| AC2 | <criterion text> | <why it did not run> | not run (#<M>) |
| AC3 | <criterion text> | <what is missing> | unmet |

Status: `met` (checked; the box is ticked at the merge), `not run (#<M>)` (needs the full gate, a GPU run or a measurement that did not run; `#<M>` is the open issue that owns the work; the box stays unticked) or `unmet`.

Closure conditions (`docs/issue-workflow.md` section 5):

- [ ] every criterion has named evidence
- [ ] the production path is real, not a stub or staged subset
- [ ] edge cases, fallback behaviour and quantitative targets are evidenced
- [ ] docs, help, schema, checkpoint and provenance are updated
- [ ] the merged-tree checks passed (lint set before and after the merge, `--no-run` when crossing: <log paths>)
- [ ] no hidden follow-up inside the original scope

Verdict: CLOSE
