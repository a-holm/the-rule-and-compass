# Setup: configure the factory once

Answers live in `../_shared/` as reference files. Every run reads them and no run asks them again. Each answer below is provisional until the owner confirms or adjusts it; on confirmation only the answer and its reasoning stay here.

## 1. Who is the workshop for, and what does a finished product look like?

Provisional answer: coding agents (Claude Code, opencode) and their orchestrator, working on `the-rule-and-compass`. A finished product is an open pull request, a merged integration branch, a gate report, a decision comment or a closed and cleaned issue, each with a `check.md` that says `pass`. Confirm or adjust.

## 2. Voice and tone: two examples that are right, one that is wrong

Provisional answer: right: "Gate on 9957f26: cargo test 1173 passed; cuda 1415 passed; rcsr-bench --tiny 27/27." Right: "Not verified: the CUDA suite." Wrong: "Everything probably passes now." Simple technical English, no em-dashes, present tense. Confirm or adjust.

## 3. Hard limits that never bend

Provisional answer: `../_shared/rules.md`. In short: one issue per worktree and PR, `Refs` not `Closes`, everything through the lock, no `pgrep -f`, no Python, one gate on the integration tip, Sonnet implements and Opus only researches. Confirm or adjust.

## 4. What is always checked before anything ships, and can a machine do it?

Provisional answer: yes. Each stage's `## Check` is a command. The gate is `../_tools/gate.sh`. The structure of this workshop is `../_tools/links.sh`. A person decides only the closure verdict, and an independent verifier proposes it. Confirm or adjust.

## 5. What already exists that runs should reuse?

Provisional answer, with the machine values the run variables take:

| Name | Value |
|---|---|
| `ROOT` | the first entry of `git worktree list` |
| Lock script | `/mnt/lager/tmp/opencode/one-full-suite.sh` |
| Pause file | `/mnt/lager/tmp/opencode/PAUSE` |
| Gate logs | `/mnt/lager/tmp/eureqa-gates/` |
| CUDA architecture | `RCSR_CUDA_ARCH=sm_86` |
| Integration branch | `integration/gpu-hardening` |
| Memory file `MEM` | the integration state file under the agent's project memory directory |
| Existing tools | `../../scripts/stacks.sh`, `../../scripts/stall-watch.sh`, `rcsr-launch`, `rcsr-rss` |

Confirm or adjust.
