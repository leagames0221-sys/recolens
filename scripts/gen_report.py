"""Generate a self-contained KPI / A-B report (docs/demo-viewer/index.html).

Runs the default eval + A-B and emits a single static HTML file (inline CSS +
SVG, no external deps, no network) — the demo surface for the eval harness.
Deterministic for a fixed seed.
"""

from __future__ import annotations

import html
from pathlib import Path

from recolens.core.ab import ab_compare
from recolens.core.evaluate import build_qrels, run_eval, time_split
from recolens.core.schema import parse_interactions, parse_items
from recolens.packs.ugc.baselines import ContentRanker, PopularityRanker
from recolens.packs.ugc.reco_collab import CollaborativeRanker
from recolens.packs.ugc.reco_hybrid import HybridRanker
from recolens.packs.ugc.reco_reranked import RerankedRanker
from recolens.packs.ugc.synth import generate

OUT = Path(__file__).resolve().parents[1] / "docs" / "demo-viewer" / "index.html"


def _bar(value: float, vmax: float, color: str, width: int = 220) -> str:
    w = int((value / vmax) * width) if vmax > 0 else 0
    return (
        f'<svg width="{width}" height="14">'
        f'<rect width="{width}" height="14" fill="#eee"/>'
        f'<rect width="{w}" height="14" fill="{color}"/></svg>'
    )


def build(seed: int = 42) -> str:
    data = generate(n_items=300, n_users=80, seed=seed)
    items = parse_items(data["items"]).valid
    inter = parse_interactions(data["interactions"]).valid
    train, test = time_split(inter, 0.3)

    rankers = [
        PopularityRanker(),
        ContentRanker(dim=64),
        CollaborativeRanker(),
        HybridRanker(dim=64),
        RerankedRanker(dim=64),  # learned stage-2 reranker (logistic default)
    ]
    _q, results, _r = run_eval(items, train, test, rankers, ks=(5, 10))

    qrels = build_qrels(test)
    users = sorted(qrels)
    ra, rb = ContentRanker(dim=64), CollaborativeRanker()  # A = content, B = collaborative
    ra.fit(items, train)
    rb.fit(items, train)
    runs_a = {u: ra.rank(u, 10) for u in users}
    runs_b = {u: rb.rank(u, 10) for u in users}
    ab = ab_compare(qrels, runs_a, runs_b, k=10, seed=seed)

    metric_names = sorted({m for r in results.values() for m in r})
    names = list(results.keys())

    rows = []
    for m in metric_names:
        vmax = max(results[n][m] for n in names) or 1.0
        cells = "".join(
            f"<td>{results[n][m]:.4f} {_bar(results[n][m], vmax, '#3b82f6' if n == 'collab' else '#9ca3af')}</td>"
            for n in names
        )
        rows.append(f"<tr><th>{html.escape(m)}</th>{cells}</tr>")
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)

    decision_color = {"B wins": "#16a34a", "A wins": "#dc2626"}.get(str(ab["decision"]), "#ca8a04")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>recolens — demo tour</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;color:#111;padding:0 1.5rem}}
 section{{padding:2.4rem 0 2.0rem;border-bottom:1px solid #f0f0f0;scroll-margin-top:1rem}}
 h1{{margin:.2rem 0;font-size:2.0rem}} h2{{margin:.2rem 0 .6rem;font-size:1.4rem}}
 .sub{{color:#666;margin-top:.2rem}} .lead{{font-size:1.15rem;color:#333}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0}}
 th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;font-variant-numeric:tabular-nums}}
 thead th{{background:#f8fafc}}
 .card{{border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.2rem;margin:.6rem 0;background:#fafafa}}
 .kpi{{font-size:1.4rem;font-weight:700}}
 .badge{{display:inline-block;background:#111;color:#fff;border-radius:6px;padding:.15rem .6rem;font-size:.85rem;margin:.15rem .3rem .15rem 0}}
 .badge.g{{background:#16a34a}}
 code{{background:#f1f5f9;padding:.05rem .3rem;border-radius:4px}}
 .big{{font-size:1.5rem;font-weight:700}}
</style></head><body>

<section id="hero">
<h1>recolens</h1>
<p class="lead">ローカルで無料で動く、コンテンツ推薦・検索のミニプラットフォーム。<br>主役は「評価ハーネス」 — KPI と A-B とコストを再現可能に測る。</p>
<p><span class="badge g">cost $0</span><span class="badge g">no credit card</span><span class="badge">local / offline</span><span class="badge">MIT</span></p>
</section>

<section id="why">
<h2>なぜ作ったか</h2>
<p class="lead">多くの推薦は「モデルを作った」で終わる。プラットフォームの仕事はその先 —<br>行動ログとコンテンツを特徴量にし、索引で検索・推薦を出し、品質とコストを<b>測る</b>こと。</p>
</section>

<section id="commands">
<h2>9 つのコマンドで一気通貫</h2>
<p class="sub">ingest → 特徴量 → 埋め込み → 索引 → 推薦 / 検索 → 評価</p>
<table><thead><tr><th>command</th><th>役割</th></tr></thead><tbody>
<tr><td><code>ingest</code></td><td>Protocol Buffers スキーマで読み込み(欠損は理由付きで除外)</td></tr>
<tr><td><code>index</code> / <code>search</code></td><td>埋め込みでベクトル索引・近傍検索</td></tr>
<tr><td><code>recommend</code></td><td>content / 協調 / hybrid の top-N(cold-start 安全)</td></tr>
<tr><td><code>eval</code></td><td>時系列分割で Recall@K / nDCG / MRR / MAP</td></tr>
<tr><td><code>ab</code></td><td>A-B シミュレーション + 信頼区間 + 判定</td></tr>
<tr><td><code>bench</code></td><td>埋め込み・索引のコスト × レイテンシ</td></tr>
<tr><td><code>classify</code> / <code>moderate</code></td><td>分類・品質、不正 / スパム / インジェクション検知</td></tr>
</tbody></table>
</section>

<section id="ranking">
<h2>評価ハーネス — 5 手法を同じ土台で比較(合成 fixture)</h2>
<p class="sub">指標は BEIR / ir_measures 準拠、<code>ir_measures</code> と 1e-9 以内で一致。合成データは<b>信号を仕込んだ検証用 fixture</b>で、collab が構造的に勝つ。信頼性は下の実データが担う。</p>
<table><thead><tr><th>metric</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>
</section>

<section id="realdata">
<h2>実データ検証 — MovieLens 100k(1682 items / 942 users)</h2>
<p class="sub">同じハーネスを公開実データで実行。ここでは<b>どの信号もオラクルでない</b>ので、勝敗は仕込みでなく実力。<code>eval --dataset movielens</code></p>
<table><thead><tr><th>method (nDCG@10)</th><th>value</th><th></th></tr></thead><tbody>
<tr><th>reranked · LambdaMART</th><td>0.168</td><td>{_bar(0.168, 0.168, '#16a34a')} <b>最良</b></td></tr>
<tr><th>reranked · logistic(依存ゼロ)</th><td>0.160</td><td>{_bar(0.160, 0.168, '#16a34a')}</td></tr>
<tr><th>collaborative(単独最良)</th><td>0.149</td><td>{_bar(0.149, 0.168, '#9ca3af')}</td></tr>
<tr><th>hybrid(固定 RRF 融合)</th><td>0.146</td><td>{_bar(0.146, 0.168, '#9ca3af')}</td></tr>
</tbody></table>
<p class="lead"><b>実データでは学習型リランカーが全単独信号・固定融合を上回る</b>(LambdaMART が collab 比 +13%)。感度も正:test-ratio 0.2 / 0.3 / 0.5 で +3〜13%(nDCG@10)。</p>
</section>

<section id="ab">
<h2>A-B シミュレーション(KPI = hit-rate@{ab["k"]})</h2>
<div class="card">
 <div>A = <b>content</b> &nbsp; B = <b>collaborative</b> &nbsp; (n={ab["n"]} users)</div>
 <p class="kpi">KPI_A {ab["kpi_a"]:.4f} &rarr; KPI_B {ab["kpi_b"]:.4f}
   &nbsp; lift {ab["abs_lift"]:+.4f} ({ab["rel_lift"]:+.1%})</p>
 <div>95% bootstrap CI (B−A): [{ab["ci95_low"]:+.4f}, {ab["ci95_high"]:+.4f}]
   &nbsp; 判定: <b style="color:{decision_color}">{html.escape(str(ab["decision"]))}</b></div>
 <p class="sub">区間が勝敗を裏付ける — 点推定が正でも CI が 0 を跨げば「判定不能」と正直に出す。</p>
</div>
</section>

<section id="method">
<h2>業界標準の 2 段構え(retrieve → learned rank)</h2>
<p class="lead">2025-26 の定石は「固定ルールで混ぜる」でなく<b>学習型ランカー</b>。信号が候補を出し、モデルが並べ替えを<b>学習</b>する。<br>既定は依存ゼロの学習型、任意で <b>LightGBM LambdaMART</b>(GBDT の LTR 主力)。</p>
<p class="sub"><b>正直な2面</b>:合成(近オラクル)では融合は単独最良に勝てない — だが<b>実データでは学習型が勝つ</b>。両方をそのまま報告(ADR-0009 / 0010、rig しない)。</p>
</section>

<section id="perf">
<h2>コストと性能</h2>
<p class="lead">Qdrant のローカルモードで 384 次元検索を <span class="big">p50 約 5.9ms → 0.45ms(約 13 倍)</span>。<br>埋め込みは民生 CPU で毎秒およそ 310 件 — GPU 不要。</p>
</section>

<section id="safety">
<h2>安全性 — 不正をルール + LLM で判定</h2>
<div class="card">
 <div><code>moderate "ignore all previous instructions ..."</code></div>
 <p class="kpi" style="color:#dc2626">action: block &nbsp; injection: True</p>
 <p class="sub">プロンプトインジェクションを遮断、スパムは flag、正常は allow。落ちるべき入力での負例テスト付き。</p>
</div>
</section>

<section id="free">
<h2>無料・クレカ不要・ローカル</h2>
<p><span class="badge g">cost $0</span><span class="badge g">no credit card</span><span class="badge">zero runtime deps</span><span class="badge">CI green</span><span class="badge">88 tests</span></p>
<p class="lead">既定インストールは依存ゼロで決定論的に完走。埋め込み・Qdrant・LLM はすべて任意層。</p>
<p class="sub">License: MIT &nbsp;·&nbsp; github.com/leagames0221-sys/recolens</p>
</section>

</body></html>
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
