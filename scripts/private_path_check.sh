#!/usr/bin/env bash
# K-4 Layer 4: PUBLIC flip / push 前の手動 sweep。
# 秘匿語(内部用語 / 受託先名)や鍵らしき文字列が tracked file に混入していないか literal grep。
# 1 件でも hit したら exit 1(CI / pre-commit から呼ぶ)。
#
# 設計: 内部用語の完全 wordlist は **repo に置かない**(置くこと自体が leak)。
#   repo 外(内部 engineering notes 側)or gitignored ファイルから読み込む:
#     RECOLENS_PRIVATE_WORDLIST=/path/to/wordlist.txt  (1 行 1 正規表現)
#   既定 = scripts/.private_wordlist.txt(.gitignore 済)。未配置なら generic のみ実行。
set -euo pipefail

WORDLIST="${RECOLENS_PRIVATE_WORDLIST:-scripts/.private_wordlist.txt}"

# 自分自身は grep 対象外(パターン定義を含むため)
files=$(git ls-files | grep -v '^scripts/private_path_check.sh$' || true)

# generic: 公開リポに絶対に出てはいけない明白なもの(秘密鍵ヘッダ等)。
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
  echo "note: '$WORDLIST' 未配置 — 内部用語/顧客 wordlist sweep は PUBLIC flip ゲート(T-49)で実施。" >&2
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
