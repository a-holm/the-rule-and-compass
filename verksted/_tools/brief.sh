#!/usr/bin/env bash
# Usage: brief.sh impl|research|verifier --issue <n> [--slug <s>] [--epic <n>] [--pr <n>] [--model sonnet|opus|fable] [--owned <paths>]
# Prints a dispatch prompt: line 1 is the no-resume line; then the values N, SLUG, W, BRANCH, BASE, G, EPIC, PR, VERKSTED (absolute path of the verksted/ it runs from);
# then, for verifier only, OUT=$VLOCAL/verifier/<PR> (the durable place of verification.md, report.md and check.md; verify-merged.sh --done never removes it);
# then "Read and follow $VERKSTED/_shared/brief.md and $VERKSTED/_templates/<template> exactly." and, for impl and research, the setup command by absolute path. For a verifier, W is its scratch worktree $WT/verifier-<PR> (verify-merged.sh creates and removes it).
# --model sonnet (default), opus or fable. An opus or fable prompt adds: use tractatus thinking, context7 and web search; a fable prompt
# adds the dispatch line `subagent_type fable-52` on stderr, never in the prompt. Every prompt ends with a line that names $VLOCAL/rules-local.md (resolved path).
# Required: impl needs --issue, --slug; research needs --issue, --slug; verifier needs --issue, --pr.
# Refuses when a required value is missing, the output still holds a `<placeholder>`, it runs from the main checkout, or its worktree lacks
# commits of origin/$BASE that touch verksted/ (no fetch: it checks the local ref; the orchestrator fetches).
# Writes: nothing.
# Exit: 0 printed; 1 refused; 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
kind=${1:-}
case "$kind" in
  -h|--help) usage; exit 0 ;;
  impl|research|verifier) shift ;;
  *) usage >&2; exit 2 ;;
esac
issue= slug= epic= pr= model=sonnet owned=
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --issue|--slug|--epic|--pr|--model|--owned)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      case "$1" in
        --issue) issue=$2 ;;
        --slug) slug=$2 ;;
        --epic) epic=$2 ;;
        --pr) pr=$2 ;;
        --model) model=$2 ;;
        --owned) owned=$2 ;;
      esac
      shift ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
case "$model" in sonnet|opus|fable) ;; *) usage >&2; exit 2 ;; esac
for v in "$issue" "$epic" "$pr"; do
  case "$v" in ''|[0-9]*) ;; *) usage >&2; exit 2 ;; esac
  case "$v" in *[!0-9]*) usage >&2; exit 2 ;; esac
done
refuse() { echo "refused: $1" >&2; exit 1; }
[ "$RCSR_ROOT" != "$ROOT" ] || refuse "run brief.sh from a worktree on the integration tip, not the main checkout (rule 61); main checkout: $ROOT"
behind=$(tools_behind "$RCSR_ROOT") || refuse "origin/$BASE not found in $RCSR_ROOT; fetch first"
[ "$behind" -eq 0 ] || refuse "this worktree is $behind commit(s) behind origin/$BASE in verksted/; merge origin/$BASE in $RCSR_ROOT (or use a current tools worktree) and rerun"
[ -n "$issue" ] || refuse "--issue is required"
case "$kind" in
  impl|research)
    [ -n "$slug" ] || refuse "--slug is required for $kind"
    [[ $slug =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || refuse "slug '$slug' is not kebab-case" ;;
  verifier)
    [ -n "$pr" ] || refuse "--pr is required for verifier" ;;
esac
case "$kind" in
  impl) template=brief.md; w=$WT/issue-$issue-$slug; branch=issue/$issue-$slug ;;
  research) template=brief-research.md; w=$WT/research-$slug; branch="none (detached read-only worktree)" ;;
  verifier) template=brief-verifier.md; w=$WT/verifier-$pr; branch="none (the PR head)"
    [ -z "$slug" ] || branch=issue/$issue-$slug ;;
esac
out=$(
  echo "You are not resumed automatically; block in your own turn until everything is done."
  echo
  echo "Kind: $kind"
  echo "N=$issue"
  echo "SLUG=${slug:-none}"
  echo "W=$w"
  echo "BRANCH=$branch"
  echo "BASE=$BASE"
  echo "G=$G"
  echo "EPIC=${epic:-none given; read it from the issue metadata, else the default epic of rules-local.md}"
  echo "PR=${pr:-none yet}"
  echo "VERKSTED=$VERKSTED"
  [ "$kind" != verifier ] || echo "OUT=$VLOCAL/verifier/$pr"
  [ "$kind" != impl ] || [ -z "$owned" ] || echo "OWNED=$owned"
  echo
  case "$kind" in
    impl) echo "Create your worktree: $VERKSTED/_tools/wt-new.sh $issue $slug" ;;
    research) echo "Create your worktree: $VERKSTED/_tools/wt-new.sh --research $slug" ;;
  esac
  echo "Read and follow $VERKSTED/_shared/brief.md and $VERKSTED/_templates/$template exactly."
  case "$model" in
    opus|fable) echo "Use tractatus thinking, context7 and web search." ;;
  esac
  echo "Read $VLOCAL/rules-local.md when it exists."
)
if printf '%s\n' "$out" | grep -qE '<[A-Za-z_][^>]*>'; then
  refuse "the prompt still holds a <placeholder>"
fi
printf '%s\n' "$out"
[ "$model" != fable ] || echo "Dispatch with subagent_type fable-52." >&2
