# 指標の一次定義(実装前突合・独自定義を作らない)

> 確認日 2026-06-30。実装(`recolens/core/metrics.py`)はこの定義に従い、IR 指標は
> `ir_measures`(trec_eval 系)と**値突合テスト**で一致を保証する(`tests/test_metrics.py`)。

## IR 指標(出典: ir_measures / trec_eval)
出典: [ir-measures measures](https://ir-measur.es/en/latest/measures.html) / [terrierteam/ir_measures](https://github.com/terrierteam/ir_measures)

| 指標 | 定義 | 既定パラメータ |
|---|---|---|
| **Recall@k** | rank k までに取得された relevant 文書の割合 = \|relevant ∩ top-k\| / \|relevant\| | rel≥1 を relevant |
| **Precision@k** | top-k のうち relevant の割合 = \|relevant ∩ top-k\| / k | cutoff 必須 |
| **AP / MAP** | 各 relevant 取得位置での precision の平均(query 平均が MAP) | rel≥1 |
| **RR / MRR** | 最初の relevant 文書の順位の逆数 1/rank | rel≥1 |
| **nDCG@k** | DCG@k / IDCG@k。**DCG@k = Σ_{i=1..k} rel(i) / log₂(i+1)**(線形 gain、dcg='log2')。IDCG = relevance 降順の理想順での DCG | dcg='log2' |

出典(nDCG 式): [evidently: nDCG](https://www.evidentlyai.com/ranking-metrics/ndcg-metric) — DCG@k = Σ rel(i)/log₂(i+1), nDCG@k = DCG@k/IDCG@k, 範囲 [0,1]。

## recsys 指標(出典: RecBole)
出典: [recbole.evaluator.metrics](https://recbole.io/docs/recbole/recbole.evaluator.metrics.html)

| 指標 | 定義 |
|---|---|
| **ItemCoverage@K** | 推薦された item の網羅率 = \|⋃_{u∈U} R̂(u)\| / \|I\| |
| **Novelty(self-information)** | 推薦リスト item の平均自己情報量。item j の自己情報 N_j = log₂(M / d_j)(M=ユーザ数、d_j=item j に interaction したユーザ数)。Novelty = 推薦 item 全体の N_j 平均 |

> 実装では Novelty を popularity ベースで `N_j = -log₂(p_j)`(p_j = d_j/M)= `log₂(M/d_j)` と等価形で算出。未観測 item は最大 novelty 側にクリップ。

## 値突合の方針(T-17)
`tests/test_metrics.py` で同一 qrels/run に対し `ir_measures.calc_aggregate([nDCG@k, AP, RR, R@k, P@k], ...)` と
`recolens.core.metrics` の集計を比較し、`abs(diff) < 1e-9` を assert(独立2実装の一致 = 確認型でない実証)。
