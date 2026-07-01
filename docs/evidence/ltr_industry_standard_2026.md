# Learning-to-rank as the industry standard (2025-2026) — cited evidence

> Primary-source scan backing ADR-0010. Verified 2026-07-01.
> Claim: the standard fix for "fixed-weight fusion can't beat a sharp single
> signal" is a **two-stage retrieve → learned-rank** pipeline with a GBDT
> LambdaMART reranker — and this is current practice, not a legacy technique.

## The two-stage architecture is the norm
- **Two-stage retrieve → rank** is the standard pipeline: a fast, low-precision
  first stage retrieves candidates (BM25 / dense vectors / late interaction), then
  a higher-precision reranker re-orders the top candidates.
  — SciRerankBench, arXiv 2025: https://arxiv.org/html/2508.08742v1
  — MultiLTR (multi-stage learning-to-rank), MDPI Information 2025:
    https://www.mdpi.com/2078-2489/16/4/308
- Recommender-system framing (candidate generation → ranking) as standard practice:
  https://recsysml.substack.com/p/early-stage-ranking-in-recommender

## The fusion layer: RRF is the baseline, learned rank is the answer
- **RRF is the recommended *starting* baseline** (parameter-light, label-free,
  robust), but is "less adaptable than learned convex combinations and prone to
  performance non-smoothness — especially under domain shift." Learning-to-rank is
  advised when top relevance matters and labels are available.
  — Elastic, "What is hybrid search": https://www.elastic.co/what-is/hybrid-search
  — Result-fusion & ranking strategies: https://apxml.com/courses/advanced-vector-search-llms/chapter-3-hybrid-search-approaches/result-fusion-ranking-strategies
  — RRF overview: https://www.emergentmind.com/topics/reciprocal-rank-fusion-rrf-algorithm

## GBDT LambdaMART is the current production workhorse
- **Gradient-boosted decision trees remain widely deployed** for ranking in 2025-2026
  due to strong performance on structured/heterogeneous features under realistic
  latency/resource constraints.
  — Unified LTR for multi-channel retrieval in large-scale e-commerce, arXiv 2026:
    https://arxiv.org/html/2602.23530
  — MoE ranking at scale (MTmixAtt), arXiv 2025: https://arxiv.org/pdf/2510.15286
  — Hybrid GBDT recommenders (LightGBM), LFDNN: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10137739/
- **LightGBM `lambdarank` objective = LambdaMART** (boosted-tree LambdaRank),
  optimizing nDCG directly.
  — LightGBM docs (LGBMRanker): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
- **LightGBM license = MIT** (Microsoft Corporation and the LightGBM developers),
  verified against the repo LICENSE — satisfies the K-1 allowlist and the
  supply-chain gate.
  — https://raw.githubusercontent.com/microsoft/LightGBM/master/LICENSE
- Frontier (text): LLM listwise rerankers (RankGPT / RankT5 / ListT5) — high
  precision, high cost; noted, not adopted for a laptop-scale free tool.

## How recolens maps to this
- Stage 1 = existing signals (content kNN, item-item collaborative, tag, popularity).
- Stage 2 = `reranked`: default **logistic** (zero-dep, deterministic) + optional
  **LightGBM LambdaMART** (`[rank]` extra).
- Measured (ADR-0010): LambdaMART beats fixed-weight RRF fusion by +18.7% nDCG@10 /
  +48% RR; it does not beat the near-oracle collaborative signal on this synthetic
  workload — reported honestly (fusion helps for complementary signals).
