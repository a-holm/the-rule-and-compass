# Run card: <issue number and title>

Copy to `../01_issue/stages/02_issue_body/output/run.md` and fill in.

## Goal

<What is true when this run is done. An outcome, not a process.>

## Scope

In:
- `<path>`: <why>

Out:
- <what is deliberately not touched, with the issue where it lands (#N)>

## Proof

One line per acceptance criterion of the issue.

- [ ] `<command>` -> <expected exit code or number>
- [ ] `<command>` -> <expected exit code or number>

## Constraints

- No Python. Experiments in Rust. Targeted tests only, through the lock.
- <files that must not change>

## Stop rule

<When to stop and report instead of improvising. Example: stop if a referenced path does not exist, or a criterion fails twice for the same reason.>

## Deviations

- <what was skipped and why. Never empty if something is missing.>
