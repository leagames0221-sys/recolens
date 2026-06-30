#!/usr/bin/env bash
# Pre-push secret/leak sweep: greps tracked files for private-key headers and any
# patterns from an optional external wordlist; exits 1 on a hit (used by CI /
# pre-commit). The wordlist itself is never committed — point at it via
# RECOLENS_PRIVATE_WORDLIST=/path/to/wordlist.txt (one regex per line). Default
# scripts/.private_wordlist.txt is gitignored; if absent, only the generic
# private-key patterns below run.
set -euo pipefail

WORDLIST="${RECOLENS_PRIVATE_WORDLIST:-scripts/.private_wordlist.txt}"

# exclude this script itself (it contains the pattern definitions)
files=$(git ls-files | grep -v '^scripts/private_path_check.sh$' || true)

# generic patterns that must never appear in a public repo (private-key headers)
patterns=(
  "BEGIN RSA PRIVATE KEY"
  "BEGIN OPENSSH PRIVATE KEY"
  "BEGIN PGP PRIVATE KEY"
)

if [ -f "$WORDLIST" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && patterns+=("$line")
  done < "$WORDLIST"
else
  echo "note: '$WORDLIST' not present — running generic patterns only." >&2
fi

hit=0
for p in "${patterns[@]}"; do
  if echo "$files" | xargs -r grep -InF "$p" 2>/dev/null; then
    echo "PRIVATE leak candidate: [$p]" >&2
    hit=1
  fi
done

if [ "$hit" -ne 0 ]; then
  echo "private_path_check FAILED" >&2
  exit 1
fi
echo "private_path_check OK"
