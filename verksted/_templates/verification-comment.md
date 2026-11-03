# Verification: <issue number and title>

Copy to `../08_close/stages/01_verifier/output/verification.md`, fill in, and post it as one issue comment. The verifier did not implement the issue and ran no cargo.

| Row | Criterion | Evidence | Status |
|---|---|---|---|
| AC1 | <criterion text> | <test name, file:line or log path> | met |
| AC2 | <criterion text> | <test name, file:line or log path> | unmet |

Closure conditions (`docs/issue-workflow.md` section 5):

- [ ] every criterion has named evidence
- [ ] the production path is real, not a stub or staged subset
- [ ] edge cases, fallback behaviour and quantitative targets are evidenced
- [ ] docs, help, schema, checkpoint and provenance are updated
- [ ] the integration gate passed (<gate log paths>)
- [ ] no hidden follow-up inside the original scope

Verdict: CLOSE
