# Brief research section: <issue number or topic>

Hard rules: never `pkill` or `killall` (kill only a pid you started, with `kill <pid>`); a wait uses `wait-for.sh`, never `pgrep -f`; no Python. A lint in `verksted_tools` fails on `pkill` and `killall` in scripts.

Used with `../_shared/brief.md`; `../_tools/brief.sh research --issue <N> --slug <slug>` prints the dispatch prompt. Replace every angle-bracket placeholder.

Goal of the whole project: <goal paragraph>
Everything merges into the base branch, never `main`.
Parked issues, which stay parked: <parked issues>

Your job is research and issue writing, not implementation. A Sonnet agent implements each issue from its body alone, so every issue is implementation-ready. No code, no branch, no commit.

Tools:
- Use `mcp__tractatus_thinking__tractatus_thinking` to structure the analysis.
- Use context7 (`mcp__context7__resolve-library-id`, `mcp__context7__query-docs`) for library and API facts (CUDA, CUB, Rust std).
- Do NOT use WebSearch or WebFetch.
- Read code on the integration branch through a read-only detached worktree: `verksted/_tools/wt-new.sh --research <slug>` from the repo root. Remove it yourself at the end: `verksted/_tools/wt-done.sh --research <slug>`. Never touch other worktrees or the repo root checkout.

Measurements (allowed, small and cheap):
- Build with `rc-run.sh build <wt> research-<slug> build --cuda -- cargo build --release --features cuda`. Run with `rc-run.sh run <wt> research-<slug> run1 -- ./target/release/rcsr ...`. Paired A/B comparisons use `ab-pair.sh`. Logs land in `$GATES` (default `/mnt/lager/tmp/rcsr-gates`).
- Each lock hold under 10 minutes. No large-store or long benchmark runs; those belong to the final run. Do not change CUDA code. The Xid count must not rise; stop if it does.
- No Python, ever. Calculations use awk, bc or a small Rust program built with `nice -n 19 rustc -O` outside the repo, under `$GATES/research-<slug>-scratch/`.
- Never pkill or killall, never git stash, never force-push. Wait with `wait-for.sh` on a log line or a done-file, never with a `pgrep -f` loop.

Issues:
- Before creating anything, read the open issues in your area (`gh issue list --state open --limit 200`, `gh issue view <n> --comments`). Do not duplicate. If an open issue already covers a finding, comment on it with your evidence and design instead of a new issue. Parked issues stay parked; if your evidence says one is needed for the goal, write that in a comment on it and say so in your report. Do not unpark it.
- Many issues of the merged work stay open by design (they close after the final test). Check `git log origin/<base> --oneline` before treating something as missing.
- Create issues with `gh issue create --label roadmap,<type>,<area> --body-file <file>`. Check `gh label list`; use `enhancement` or `bug`, plus `area:gpu`, `area:perf`, `area:testing` and others that exist. The repo has no milestones and no project board; write "none" for those fields.
- Body structure, from `issue-body.md`: metadata table (planned dates YYYY-MM-DD, counted from today), Goal and background (measured evidence and code paths), Scope (every out-of-scope item names its issue), Design proposal, Acceptance criteria (measurable), Test plan, Dependencies. Run `verksted/_tools/issue-lint.sh <body.md>` before creating it. Create blockers first, so they get lower numbers.
- Plain technical English, short sentences, no em-dashes, no AI or session references, no provenance notes. Measured numbers go in the issue body, never in code comments.
- Issue and PR comments state only facts that are true now and that you verified. No predictions or plans stated as facts; write "not run", "open" or "not verified".

Notes go to `arbeidsmappe/crypto-rcsr/research/<slug>/` (`question.md`, `method.md`, `decision.md`; starters `research-question.md`, `research-decision.md`). Do not commit them.

Final report (last message): the issues you created (number, title, priority, size, blocked by), the comments you added to existing issues, the top three expected gains with their evidence, and anything that needs the project lead's decision.
