# 性能・コスト改善 narrative(T-41)

`recolens bench --all` の実測(300 items / 50 queries / k=10 / CPU、2026-06-30)。
生成物: [BENCHMARK.generated.md](../BENCHMARK.generated.md)。latency は wall-clock(run 毎に変動)。

| config | dim | embed ms/text | search p50 ms | tag-match@10 | mem(KB est) |
|---|---|---|---|---|---|
| hash + memory | 64 | 0.026 | 0.83 | 0.956 | 75 |
| e5-small + memory | 384 | 7.238 | 5.888 | 1.00 | 450 |
| **e5-small + qdrant** | 384 | 6.976 | **0.45** | 1.00 | 450 |
| hash + qdrant | 64 | 0.035 | 0.43 | 0.956 | 75 |

## 読み解き(案件「性能およびコストの改善 / 推論効率化」)
1. **品質↔コスト**: hash は埋め込み実質ゼロコスト・依存ゼロ(高速反復の既定)。e5-small は埋め込みコスト増(~7ms/text)と引き換えに**トピック検索品質が上**(tag-match@10 0.956→1.00)。ただし合成データの recsys nDCG では literal トークン一致が支配し hash 優位(指標で結論が変わる好例・正直所見は `embedding_spike_b1.md`)。
2. **検索レイテンシの改善**: 384 次元では in-memory ブルートフォース p50 **5.89ms** に対し **Qdrant(ANN/最適化済)p50 0.45ms = 約 13 倍高速**。次元が上がるほど Qdrant の優位が出る = スケール時の正しい選択。
3. **意思決定**: 「品質・レイテンシ・コスト」の予算で構成を選ぶ。小規模/低レイテンシ重視=hash+memory、品質重視=e5+qdrant。同一 CLI・同一指標で横並び比較できることが本ツールの価値。

> 数値はすべて `recolens bench` で再現可能・民生 CPU(GPU 不要、R-PERF-2)。
