#!/usr/bin/env bash
# Usage: issue-lint.sh <body.md | -> [--offline] [--number <n>] [--allow-blocker <n>]...
# Checks an issue body against the structure of verksted/_templates/issue-body.md. Prints one
# `FAIL: <rule>` line per problem and `OK` when none. `-` reads the body from stdin.
# --offline skips the gh checks (labels exist, blockers exist). --number <n> requires every blocker
# to be lower than <n>; --allow-blocker <n> (repeatable, or a comma list) exempts a known exception.
# Online checks use `gh` read-only in the current repository; RCSR_GH overrides the command.
# Writes: nothing.
# Exit: 0 clean; 1 problems; 2 usage or unreadable input.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
GH=${RCSR_GH:-gh}

src=''
offline=0
number=''
allow=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --offline) offline=1 ;;
    --number|--allow-blocker)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      opt=$1
      shift
      case $opt in
        --number)
          [[ $1 =~ ^[0-9]+$ ]] || { echo "issue-lint: --number needs an integer" >&2; exit 2; }
          number=$1 ;;
        *)
          IFS=',' read -r -a parts <<< "$1"
          for p in "${parts[@]}"; do
            p=${p#\#}
            [[ $p =~ ^[0-9]+$ ]] || { echo "issue-lint: --allow-blocker needs integers" >&2; exit 2; }
            allow+=("$p")
          done ;;
      esac ;;
    -) [ -z "$src" ] || { usage >&2; exit 2; }; src=- ;;
    -*) usage >&2; exit 2 ;;
    *) [ -z "$src" ] || { usage >&2; exit 2; }; src=$1 ;;
  esac
  shift
done
[ -n "$src" ] || { usage >&2; exit 2; }
if [ "$src" = - ]; then
  mapfile -t L < <(tr -d '\r')
else
  [ -r "$src" ] && [ -f "$src" ] || { echo "issue-lint: cannot read $src" >&2; exit 2; }
  mapfile -t L < <(tr -d '\r' < "$src")
fi

problems=()
fail() { problems+=("$1"); }
trim() {
  local s=$1
  s=${s#"${s%%[![:space:]]*}"}
  s=${s%"${s##*[![:space:]]}"}
  printf '%s' "$s"
}
in_list() {
  local x=$1
  shift
  local y
  for y in "$@"; do [ "$x" = "$y" ] && return 0; done
  return 1
}

n=${#L[@]}
fence=0
heads=()
head_at=()
for ((i = 0; i < n; i++)); do
  line=${L[i]}
  if [[ $line =~ ^[[:space:]]*(\`\`\`|~~~) ]]; then
    fence=$((1 - fence))
    continue
  fi
  [ "$fence" -eq 0 ] || continue
  if [[ $line =~ ^##[[:space:]]+(.*)$ ]]; then
    heads+=("$(trim "${BASH_REMATCH[1]}")")
    head_at+=("$i")
  fi
done

# Metadata table
fields=(Milestone Epic/Area Priority Size Estimate "Planned start" "Planned end" "Blocked by" Labels)
declare -A V=()
rows=()
first=-1
for ((i = 0; i < n; i++)); do
  if [ -n "$(trim "${L[i]}")" ]; then first=$i; break; fi
done
if [ "$first" -lt 0 ] || ! [[ ${L[first]} =~ ^\|[[:space:]]*Field[[:space:]]*\|[[:space:]]*Value[[:space:]]*\|[[:space:]]*$ ]]; then
  fail "table-missing: the body must start with a '| Field | Value |' table"
else
  i=$((first + 1))
  if [ "$i" -lt "$n" ] && [[ ${L[i]} =~ ^\|[-:\ \|]+$ ]]; then i=$((i + 1)); fi
  for (( ; i < n; i++)); do
    [[ ${L[i]} =~ ^\| ]] || break
    IFS='|' read -r _ f v _ <<< "${L[i]}"
    f=$(trim "$f")
    v=$(trim "$v")
    rows+=("$f")
    V[$f]=$v
  done
  seen=()
  for f in "${fields[@]}"; do
    if in_list "$f" "${rows[@]}"; then seen+=("$f"); else fail "field-missing: $f"; fi
  done
  for f in "${rows[@]}"; do
    in_list "$f" "${fields[@]}" || fail "field-unknown: $f"
  done
  if [ "${#seen[@]}" -eq "${#fields[@]}" ]; then
    ordered=()
    for f in "${rows[@]}"; do in_list "$f" "${fields[@]}" && ordered+=("$f"); done
    if [ "${ordered[*]}" != "${fields[*]}" ]; then
      fail "field-order: expected ${fields[*]}"
    fi
  fi

  [ -z "${V[Priority]+x}" ] || [[ ${V[Priority]} =~ ^P[0-3]$ ]] || fail "priority-invalid: '${V[Priority]}' is not P0 to P3"
  [ -z "${V[Size]+x}" ] || [[ ${V[Size]} =~ ^(XS|S|M|L|XL)$ ]] || fail "size-invalid: '${V[Size]}' is not XS, S, M, L or XL"
  [ -z "${V[Estimate]+x}" ] || [[ ${V[Estimate]} =~ ^[0-9]+(\.[0-9]+)?[[:space:]]*d$ ]] || fail "estimate-invalid: '${V[Estimate]}' is not a number and d"
  dates_ok=1
  for f in "Planned start" "Planned end"; do
    [ -n "${V[$f]+x}" ] || { dates_ok=0; continue; }
    d=${V[$f]}
    if ! [[ $d =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || [ "$(date -d "$d" +%F 2>/dev/null)" != "$d" ]; then
      fail "date-invalid: $f '$d' is not YYYY-MM-DD"
      dates_ok=0
    fi
  done
  if [ "$dates_ok" -eq 1 ] && [[ ${V[Planned start]} > ${V[Planned end]} ]]; then
    fail "date-order: Planned start ${V[Planned start]} is after Planned end ${V[Planned end]}"
  fi
  if [ -n "${V[Blocked by]+x}" ]; then
    bb=${V[Blocked by]}
    if [ "${bb,,}" != none ] && ! [[ $bb =~ ^#[0-9]+([,[:space:]]+#[0-9]+)*$ ]]; then
      fail "blocked-by-invalid: '$bb' is not 'none' or a list of #N"
    fi
  fi
fi

# Headings
want=("Goal and background" "Scope" "Design proposal" "Acceptance criteria" "Test plan" "Dependencies")
pos=()
for w in "${want[@]}"; do
  found=-1
  for k in "${!heads[@]}"; do
    h=${heads[k]}
    if [ "$h" = "$w" ] || { [ "$w" = "Design proposal" ] && [[ $h == "Design proposal"* ]]; }; then
      found=$k
      break
    fi
  done
  if [ "$found" -lt 0 ]; then
    fail "heading-missing: ## $w"
  else
    pos+=("$found")
  fi
done
prev=-1
for p in "${pos[@]}"; do
  if [ "$p" -le "$prev" ]; then
    fail "heading-order: the ## headings must follow ${want[*]}"
    break
  fi
  prev=$p
done

# Lines of one section, one per output line (empty when the heading is absent).
section() {
  local w=$1 k start end i
  for k in "${!heads[@]}"; do
    if [ "${heads[k]}" = "$w" ] || { [ "$w" = "Design proposal" ] && [[ ${heads[k]} == "Design proposal"* ]]; }; then
      start=${head_at[k]}
      end=$n
      if [ $((k + 1)) -lt "${#head_at[@]}" ]; then end=${head_at[k + 1]}; fi
      for ((i = start + 1; i < end; i++)); do printf '%s\n' "${L[i]}"; done
      return
    fi
  done
}

if in_list Scope "${heads[@]}"; then
  mapfile -t S < <(section Scope)
  has_in=0
  has_out=0
  for line in "${S[@]}"; do
    if [[ $line =~ ^In\ scope ]]; then
      has_in=1
    elif [[ $line =~ ^Out\ of\ scope ]]; then
      has_out=1
    elif [ "$has_out" -eq 1 ] && [[ $line =~ ^[[:space:]]*[-*][[:space:]] ]] && ! [[ $line =~ \#[0-9]+ ]]; then
      fail "scope-out-without-issue: ${line:0:70}"
    fi
  done
  [ "$has_in" -eq 1 ] || fail "scope-in-missing: no 'In scope' line"
  [ "$has_out" -eq 1 ] || fail "scope-out-missing: no 'Out of scope' line"
fi

if in_list "Acceptance criteria" "${heads[@]}"; then
  if ! section "Acceptance criteria" | grep -qE '^[[:space:]]*-[[:space:]]\[[ xX]\][[:space:]]*[^[:space:]]'; then
    fail "acceptance-missing: no '- [ ]' criterion"
  fi
fi

dep_nums=()
if in_list Dependencies "${heads[@]}"; then
  mapfile -t D < <(section Dependencies)
  has_bb=0
  for line in "${D[@]}"; do
    if [[ $line == *"Blocked by:"* ]]; then
      has_bb=1
      rest=${line#*Blocked by:}
      while [[ $rest =~ \#([0-9]+) ]]; do
        dep_nums+=("${BASH_REMATCH[1]}")
        rest=${rest#*"${BASH_REMATCH[0]}"}
      done
    fi
  done
  [ "$has_bb" -eq 1 ] || fail "dependencies-blocked-by-missing: no 'Blocked by:' line"
fi

# Forbidden text
body=$(printf '%s\n' "${L[@]}")
if grep -qiE '(closes|fixes|resolves)[[:space:]]+#' <<< "$body"; then
  fail "closing-keyword: Closes, Fixes or Resolves followed by #N"
fi
if grep -qF "$(printf '\xe2\x80\x94')" <<< "$body"; then
  fail "em-dash: replace it with a comma, colon or new sentence"
fi

# Blockers
blockers=()
if [ -n "${V[Blocked by]+x}" ]; then
  rest=${V[Blocked by]}
  while [[ $rest =~ \#([0-9]+) ]]; do
    blockers+=("${BASH_REMATCH[1]}")
    rest=${rest#*"${BASH_REMATCH[0]}"}
  done
fi
blockers+=("${dep_nums[@]}")
uniq_blockers=()
for b in "${blockers[@]}"; do
  in_list "$b" "${uniq_blockers[@]}" || uniq_blockers+=("$b")
done
if [ -n "$number" ]; then
  for b in "${uniq_blockers[@]}"; do
    if [ "$b" -ge "$number" ] && ! in_list "$b" "${allow[@]}"; then
      fail "blocker-order: blocker #$b is not lower than #$number"
    fi
  done
fi

if [ "$offline" -eq 0 ]; then
  if names=$("$GH" label list --limit 1000 --json name --jq '.[].name' 2>/dev/null); then
    mapfile -t known <<< "$names"
    IFS=',' read -r -a labels <<< "${V[Labels]:-}"
    for lab in "${labels[@]}"; do
      lab=$(trim "${lab//\`/}")
      [ -z "$lab" ] || in_list "$lab" "${known[@]}" || fail "label-unknown: $lab"
    done
  else
    fail "gh-unavailable: gh label list failed"
  fi
  for b in "${uniq_blockers[@]}"; do
    "$GH" issue view "$b" --json number > /dev/null 2>&1 || fail "blocker-unknown: #$b does not exist"
  done
fi

if [ "${#problems[@]}" -eq 0 ]; then
  echo OK
  exit 0
fi
for p in "${problems[@]}"; do echo "FAIL: $p"; done
exit 1
