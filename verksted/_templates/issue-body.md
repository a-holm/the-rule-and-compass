# Issue body: <type(area): title>

Copy to a temporary file, fill in, and pass it to `gh issue create --body-file`. Every issue is implementation-ready: a developer starts coding without archaeology.

| Field | Value |
|---|---|
| Milestone | none (the repo has no milestones) |
| Epic/Area | <area:*> |
| Priority | <P0 to P3> |
| Size | <XS to XL> |
| Estimate | <days> |
| Planned start | <YYYY-MM-DD> |
| Planned end | <YYYY-MM-DD> |
| Blocked by | <#N list or none> |
| Labels | <roadmap, area:*, type> |

## Goal and background

<Why the issue exists, what it unlocks, the governing docs by section.>

## Scope

In scope:
- <item>

Out of scope:
- <item> (#<N>)

## Design proposal (proposal, not mandate)

<Concrete sketches, schemas, CLI transcripts, flag tables. Numbered decisions with reasoning. Edge cases.>

## Acceptance criteria

- [ ] <objectively measurable: row count, exit code, test name, latency number>

## Test plan

- <which test proves which criterion>

## Dependencies

- Blocked by: #<N> <title>
- Blocks: #<M> <title>
