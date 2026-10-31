# Pull request body: <title>

Copy to `../01_issue/stages/06_pr/output/pr.md`, fill in, and pass it to `gh pr create --body-file`.

Refs #<N>

## What changed

- <file or area>: <change>

## Acceptance criteria

- [x] <criterion copied from the issue> (<test name or log path>)
- [ ] <criterion not met yet>

## Verification

- Prove red on `<sha>`: `<command>` -> <failing test names>
- Green: `<command>` -> <counts>, log `<path>`
- Not verified: <what this PR did not run>

## Deviations

- <deviation from the issue design and the reason, or "none">
