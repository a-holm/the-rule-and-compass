#!/usr/bin/env bash
# Usage: leak-scan.sh [--list <file>] [--diff <base>..<head>]
# Fails when a tracked file's path or text contains a banned word, case-insensitive, fixed strings.
# The word list lives outside git: --list, else $RCSR_BANNED_WORDS, else ~/.config/rcsr/banned-words;
# one word per line; blank lines and lines starting with # are ignored.
# Default: every file of `git -C "$RCSR_ROOT" ls-files`; binary files (grep -I) are skipped.
# --diff: only the added lines of `git diff <base>..<head>` and the commit messages of `git log <base>..<head>`.
# Hits print as `<path>:<line>: banned word <k>` (k = line number in the list), never the word itself.
# Writes: nothing.
# Exit: 0 clean; 1 hits; 3 no list (prints `skip: no banned-word list at <path>`); 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
list=""
range=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --list) [ $# -ge 2 ] || { usage >&2; exit 2; }; list=$2; shift 2 ;;
    --diff) [ $# -ge 2 ] || { usage >&2; exit 2; }; range=$2; shift 2 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[ -n "$list" ] || list=${RCSR_BANNED_WORDS:-$HOME/.config/rcsr/banned-words}
if [ ! -f "$list" ]; then
  echo "skip: no banned-word list at $list"
  exit 3
fi

words=()
idx=()
n=0
while IFS= read -r w || [ -n "$w" ]; do
  n=$((n + 1))
  w=${w%$'\r'}
  case "$w" in ''|'#'*) continue ;; esac
  words+=("$w")
  idx+=("$n")
done < "$list"
if [ "${#words[@]}" -eq 0 ]; then
  echo "skip: no banned-word list at $list"
  exit 3
fi

hits=0
report() { echo "$1:$2: banned word $3"; hits=$((hits + 1)); }

if [ -z "$range" ]; then
  while IFS= read -r -d '' f; do
    p=$RCSR_ROOT/$f
    [ -f "$p" ] || continue
    for i in "${!words[@]}"; do
      w=${words[$i]}
      k=${idx[$i]}
      if [[ "${f,,}" == *"${w,,}"* ]]; then report "$f" 0 "$k"; fi
      while IFS=: read -r ln _; do
        report "$f" "$ln" "$k"
      done < <(grep -I -i -n -F -e "$w" -- "$p" 2>/dev/null)
    done
  done < <(git -C "$RCSR_ROOT" ls-files -z)
else
  case "$range" in *..*) ;; *) usage >&2; exit 2 ;; esac
  file=""
  ln=0
  while IFS= read -r line; do
    case "$line" in
      '+++ b/'*) file=${line#+++ b/}; continue ;;
      '+++ '*) file=""; continue ;;
      '--- '*) continue ;;
      '@@ '*)
        h=${line#*+}; h=${h%% *}; ln=${h%%,*}; ln=$((ln - 1)); continue ;;
      '+'*) ln=$((ln + 1)) ;;
      ' '*) ln=$((ln + 1)); continue ;;
      *) continue ;;
    esac
    text=${line#+}
    for i in "${!words[@]}"; do
      w=${words[$i]}
      if [[ "${text,,}" == *"${w,,}"* ]]; then report "${file:-?}" "$ln" "${idx[$i]}"; fi
    done
  done < <(git -C "$RCSR_ROOT" diff --no-color -U0 "$range")
  while IFS= read -r -d '' rec; do
    sha=${rec%%$'\n'*}
    body=${rec#*$'\n'}
    for i in "${!words[@]}"; do
      w=${words[$i]}
      if [[ "${body,,}" == *"${w,,}"* ]]; then report "commit-$sha" 0 "${idx[$i]}"; fi
    done
  done < <(git -C "$RCSR_ROOT" log -z --format='%H%n%B' "$range")
fi

if [ "$hits" -gt 0 ]; then
  exit 1
fi
exit 0
