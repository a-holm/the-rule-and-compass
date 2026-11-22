---
type: rules
description: Which model plans and which executes, by default, and how to dispatch each tier.
---

# Models: who plans and who executes

These are defaults. An instruction from the project lead for a task replaces them for that task.

| Tier | Work | Dispatch |
|---|---|---|
| Fable 5.2 | plans cases that are extraordinarily hard or demanding: designs with many interacting constraints, proofs, plans a wrong version of which would cost weeks | `subagent_type: fable-52` |
| Opus | plans everything else: work-package splits, research synthesis, issue design, pre-registration design | `model: opus` |
| Sonnet | executes: implementation, verifiers, verification, store builds, measurements, issue filing, salvage | `model: sonnet` |

Rules
- Never dispatch `model: fable`: that name is an older version. If a newer Fable agent type exists, use the newest and update this file.
- Fable plans; it does not execute unless the project lead asks.
- Set the model or agent type on every dispatch. `_tools/brief.sh --model sonnet|opus|fable` prints the matching line.
- A planning or research agent structures its reasoning with tractatus thinking, uses context7 for library and tool facts, and searches the web whenever that helps.
- Everything that can be delegated is delegated. The main session orchestrates, decides and merges.
- A Sonnet task that fails twice on one criterion, or reports an unclear choice, gets a sharper brief on a fresh Sonnet agent; the analysis of why it failed goes to Opus.
- An agent that judges a task impossible reports the exact blocker with evidence; an Opus agent double-checks before the verdict is accepted.
- A planning agent that stops on an API error is rerun on the next tier down.
- While a long GPU run is active, keep the agents that build to `CAMPAIGN_BUILD_CAP` of `machine.md`.

Source: project lead, 2026-11-22 (three tiers); 2026-11-01 (delegate, double-check); 2026-11-09 (tractatus thinking, context7, web search).
