# recolens — Prior-Art Catalog(一次確認済、cross-session reuse 用)

> 一次確認した source を記録し再 fetch を避ける。確認日 2026-06-30。
> 全件 permissive ライセンス・大手/学術メンテで、依存セキュリティ確認を通過(red flag なし)。

| ひな形 | URL | ライセンス | 抽出するコア要素 | 真似る/自作 |
|---|---|---|---|---|
| Microsoft Recommenders | https://github.com/recommenders-team/recommenders | MIT | 5タスク構造(Prepare→Model→Evaluate→Select/Optimize→Operationalize)・オフライン指標の作法 | 構造を真似る・実装は自作 |
| RecBole | https://github.com/RUCAIBox/RecBole / https://github.com/RUCAIBox/RecBole2.0 | MIT | config 駆動で多アルゴリズムを同一 IF・再現可能設計 | 設計思想を真似る |
| Qdrant | https://github.com/qdrant/qdrant / https://qdrant.tech/documentation/quickstart/ | Apache-2.0 | Docker ローカル完結・機能ゲート無しベクトル索引 | provider として採用(任意層) |
| BEIR | https://github.com/beir-cellar/beir | Apache-2.0 | IR 評価(Precision/Recall/MAP/MRR/nDCG)の標準定義 | 指標定義を準拠 |
| ir_measures / pytrec_eval | https://github.com/terrierteam/ir_measures / https://ir-measur.es/ | MIT(pytrec_eval) | TREC 互換の指標計算・標準名 | 値突合の基準に使用 |
| multilingual-e5 / BGE-m3 | https://huggingface.co/BAAI/bge-m3 / https://sbert.net/ | 両方 MIT | 日本語対応・無料埋め込み・`query:`/`passage:` prefix・dense+lexical | provider として採用(ローカル) |
| Two-Tower retrieval | https://docs.cloud.google.com/architecture/implement-two-tower-retrieval-large-scale-candidate-generation / https://www.uber.com/en/blog/innovative-recommendation-applications-using-two-tower-embeddings/ | 公開設計(コードでなく設計) | user塔/item塔→共有空間→ANN・時系列split→recall@k・索引更新 | 設計を縮約・自作 |

## 補足(stack 判断)
- **rye / uv**: https://rye.astral.sh/ / https://docs.astral.sh/uv/ — Astral が rye を引取り、**rye は 2026 凍結(更新・セキュリティ更新なし)**、uv が後継。「rye(uv backend)」は **uv 直接採用で互換**。
- **pytrec_eval の注意**: 本体 MIT だが trec_eval は別ライセンス。recolens は ir_measures/pytrec_eval を **値突合の参照**として dev/test extras に置き、runtime 必須にはしない。

## 確認方法
各 GitHub/公式 docs を直接参照(2026-06-30)。API 詳細は実装時に公式 docs で再確認。
