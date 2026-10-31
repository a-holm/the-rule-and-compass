# Brief ticket section: <issue number and title>

Append to a copy of `../_shared/brief.md` as `../03_orchestrate/stages/02_brief/output/brief-<N>.md`. Replace every angle-bracket placeholder.

Ticket: #<N>, slug <slug>, model <sonnet|opus>.
Worktree: <absolute path, for example the repo root plus /worktrees/issue-N-slug>
Base: <integration branch or main>; branch issue/<N>-<slug>. Set N, W and G as in `commands.md`.
Owned files: <paths this agent may change>
Forbidden files: <paths other agents own>
Design: <link to the decision comment, or "the issue body is the design">
Acceptance criteria: <copy the checkbox list>
Not to touch: parked issues and everything outside the owned files.
