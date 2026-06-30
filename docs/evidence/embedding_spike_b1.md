# B-1 spike — local embedding latency + quality (Phase 3)

確認日 2026-06-30。民生 laptop CPU(GPU 不使用)。モデル = `intfloat/multilingual-e5-small`
(rev `614241f6…`、MIT、safetensors)。

## レイテンシ / スループット(CPU、batch=32、384-dim)
- 初回 load + download: 約 64s(以後はローカルキャッシュ、オフライン)。
- per-text 埋め込み: **p50 ≈ 3.25 ms / mean ≈ 3.50 ms**。
- スループット: **約 310 texts/s**(200 件 0.65s)。
- query は L2 正規化(norm=1.0)を確認。

> 民生 CPU で実用速度(GPU 不要)= 「システムパフォーマンス考慮」「推論効率化」の実測根拠。

## 検索品質: hash 近似 vs 実 e5(同一 eval、300 items / 80 users / seed 42 / 時系列分割)

> ⚠️ **重要な正直所見(2026-06-30 再測、現行 synth でデータ再設計後)**: 合成データの
> content 信号は「テーマ語彙の literal トークン重複」であり、トークンを hash する決定論
> 埋め込みの方がこの信号に合致するため、**semantic な e5 はむしろ不利**になる。

| ranker | nDCG@10 |
|---|---|
| content-hash(zero-dep 決定論) | **0.2005** |
| content-e5(実埋め込み) | 0.1788 |
| 差分 | **−11%**(e5 が下) |

> 教訓: **合成 lexical データでは埋め込み品質を公正に評価できない**(literal トークン
> 一致が支配する)。e5/BGE の真価は実テキストでのみ測れる。よって README では「e5 が
> hash に勝つ」を看板にしない(旧版の +25% 主張は前世代 synth 由来で撤回)。ハーネスは
> `--extra embed` で実 e5/BGE を回せること自体が成果(latency 実測は上記の通り実用域)。

## 再現
```
uv sync --extra embed --extra vector
recolens search "dragon wizard kingdom magic quest" --embed local --backend qdrant --k 5
```
評価の数値は `tests/`(embed extra 導入時のみ実行、CI では skip)+ 本 spike スクリプトで再現。
