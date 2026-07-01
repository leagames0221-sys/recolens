"""recolens CLI entrypoint.

Argument parser + subcommand routing are wired so that `recolens --help` and
`recolens <cmd> --help` work; each subcommand has its own handler.
"""

from __future__ import annotations

import argparse
import sys

from recolens import __version__

# (subcommand, help) — keep in sync with the WBS / design doc.
SUBCOMMANDS: list[tuple[str, str]] = [
    ("ingest", "合成/入力データを protobuf スキーマで読み込む (R-CORE-1)"),
    ("index", "アイテムを埋め込み、ベクトル索引を構築する"),
    ("search", "埋め込み近傍探索で top-K を返す (R-SEARCH-1)"),
    ("recommend", "ユーザに top-N を推薦する (R-RECO-1)"),
    ("eval", "時系列分割でオフライン指標を算出する (R-EVAL-1)"),
    ("ab", "2 構成のオフライン A-B シミュレーションを行う (R-EVAL-3)"),
    ("bench", "埋め込み/ANN のコスト×レイテンシをベンチする (R-PERF-1)"),
    ("classify", "コンテンツのカテゴリ/品質を判定する (R-LLM-1)"),
    ("moderate", "不正/スパム/インジェクションを判定する (R-LLM-2)"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recolens",
        description="Local content recommendation & search mini-platform with an eval harness.",
    )
    parser.add_argument("--version", action="version", version=f"recolens {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    implemented = {
        "ingest": _register_ingest,
        "eval": _register_eval,
        "ab": _register_ab,
        "search": _register_search,
        "recommend": _register_recommend,
        "classify": _register_classify,
        "moderate": _register_moderate,
        "bench": _register_bench,
    }
    for name, help_text in SUBCOMMANDS:
        sp = sub.add_parser(name, help=help_text)
        if name in implemented:
            implemented[name](sp)
        else:
            sp.set_defaults(_handler=_not_implemented(name))
    return parser


def _register_ingest(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--n-items", type=int, default=300, help="synthetic item count")
    sp.add_argument("--n-users", type=int, default=80, help="synthetic user count")
    sp.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    sp.add_argument("--dim", type=int, default=64, help="embedding dimension")
    sp.add_argument("--out", default=None, help="run dir (default: runs/<ts>)")
    sp.set_defaults(_handler=_ingest)


def _ingest(args: argparse.Namespace) -> int:
    # Imports are local so `recolens --help` stays fast and dependency-light.
    from recolens.core import manifest
    from recolens.core.embedding import DeterministicHashEmbed
    from recolens.core.pipeline import build_item_index
    from recolens.core.schema import parse_interactions, parse_items, parse_users
    from recolens.core.vector_index import InMemoryFlatIndex
    from recolens.packs.ugc.synth import generate

    data = generate(n_items=args.n_items, n_users=args.n_users, seed=args.seed)
    items = parse_items(data["items"])
    users = parse_users(data["users"])
    inter = parse_interactions(data["interactions"])

    embedder = DeterministicHashEmbed(dim=args.dim)
    index = build_item_index(items.valid, embedder, InMemoryFlatIndex())

    run_dir = manifest.new_run_dir() if args.out is None else args.out
    path = manifest.write_manifest(
        run_dir,
        {
            "stage": "ingest",
            "seed": args.seed,
            "embedder": "DeterministicHashEmbed",
            "dim": args.dim,
            "counts": {
                "items_valid": items.n_valid,
                "items_rejected": items.n_rejected,
                "users_valid": users.n_valid,
                "users_rejected": users.n_rejected,
                "interactions_valid": inter.n_valid,
                "interactions_rejected": inter.n_rejected,
                "indexed_items": len(index),
            },
        },
    )

    print(
        f"ingested: items={items.n_valid}(rej {items.n_rejected}) "
        f"users={users.n_valid}(rej {users.n_rejected}) "
        f"interactions={inter.n_valid}(rej {inter.n_rejected}) "
        f"indexed={len(index)}"
    )
    print(f"manifest: {path}")
    return 0


def _register_eval(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--n-items", type=int, default=300)
    sp.add_argument("--n-users", type=int, default=80)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--dim", type=int, default=64)
    sp.add_argument("--test-ratio", type=float, default=0.3)
    sp.add_argument("--ks", default="5,10", help="comma-separated cutoffs")
    sp.add_argument("--split", choices=["time"], default="time")
    sp.add_argument(
        "--dataset",
        choices=["synth", "movielens"],
        default="synth",
        help="synth=deterministic fixture (default); movielens=real ml-100k (fetch it first)",
    )
    sp.add_argument(
        "--reranker",
        choices=["logistic", "lightgbm"],
        default="logistic",
        help="stage-2 model for 'reranked' (logistic=default/zero-dep; lightgbm=[rank] extra)",
    )
    sp.add_argument("--out", default=None, help="run dir (default: runs/<ts>)")
    sp.set_defaults(_handler=_eval)


def _eval(args: argparse.Namespace) -> int:
    from recolens.core import manifest
    from recolens.core.evaluate import run_eval, time_split
    from recolens.core.schema import parse_interactions, parse_items
    from recolens.packs.ugc.synth import generate

    ks = tuple(int(x) for x in args.ks.split(","))
    dataset = getattr(args, "dataset", "synth")
    if dataset == "movielens":
        from recolens.datasets.movielens import load as load_movielens

        data = load_movielens()
    else:
        data = generate(n_items=args.n_items, n_users=args.n_users, seed=args.seed)
    items = parse_items(data["items"]).valid
    interactions = parse_interactions(data["interactions"]).valid

    train, test = time_split(interactions, test_ratio=args.test_ratio)
    reranker = getattr(args, "reranker", "logistic")
    rankers = [_make_ranker(m, args.dim, reranker) for m in _METHODS]
    _qrels, results, _runs = run_eval(items, train, test, rankers, ks=ks)

    metric_names = sorted({m for r in results.values() for m in r})
    _print_eval_table(results, metric_names)

    run_dir = manifest.new_run_dir() if args.out is None else args.out
    path = manifest.write_manifest(
        run_dir,
        {
            "stage": "eval",
            "dataset": dataset,
            "split": args.split,
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "ks": list(ks),
            "n_test_users": len({t.user_id for t in test}),
            "results": results,
        },
    )
    print(f"\nmanifest: {path}")
    return 0


def _build_embedder(name: str, dim: int):
    from recolens.core.embedding import DeterministicHashEmbed

    if name == "local":
        from recolens.providers.embed_local import LocalSentenceTransformerEmbed

        return LocalSentenceTransformerEmbed()
    return DeterministicHashEmbed(dim=dim)


def _build_index(backend: str, dim: int):
    from recolens.core.vector_index import InMemoryFlatIndex

    if backend == "qdrant":
        try:
            from recolens.providers.vector_qdrant import QdrantIndex

            return QdrantIndex(dim), "qdrant"
        except Exception as e:  # qdrant-client missing or server unreachable
            sys.stderr.write(f"qdrant unavailable ({e}); falling back to in-memory\n")
            return InMemoryFlatIndex(), "memory(fallback)"
    return InMemoryFlatIndex(), "memory"


def _register_search(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("query", help="search query text")
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--embed", choices=["hash", "local"], default="hash")
    sp.add_argument("--backend", choices=["memory", "qdrant"], default="memory")
    sp.add_argument("--n-items", type=int, default=200)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--dim", type=int, default=64, help="dim for hash embedder")
    sp.set_defaults(_handler=_search)


def _search(args: argparse.Namespace) -> int:
    from recolens.core.schema import item_text, parse_items
    from recolens.packs.ugc.synth import generate

    data = generate(n_items=args.n_items, n_users=1, seed=args.seed)
    items = parse_items(data["items"]).valid

    embedder = _build_embedder(args.embed, args.dim)
    index, backend = _build_index(args.backend, embedder.dim)
    index.add([it.item_id for it in items], embedder.embed([item_text(it) for it in items]))

    qvec = embedder.embed_queries([args.query])[0]
    hits = index.search(qvec, args.k)
    by_id = {it.item_id: it for it in items}

    print(f"query: {args.query!r}  (embed={args.embed}, backend={backend}, dim={embedder.dim})")
    for rank, (iid, score) in enumerate(hits, 1):
        it = by_id.get(iid)
        tags = ",".join(it.tags) if it else ""
        title = it.title if it else ""
        print(f"{rank:2}. {score:.3f}  {iid}  [{tags}]  {title}")
    return 0


_METHODS = ("popularity", "content", "collab", "hybrid", "reranked")


def _make_reranker_model(kind: str):
    """logistic = dependency-free default (CI); lightgbm = LambdaMART ([rank] extra)."""
    if kind == "lightgbm":
        from recolens.providers.rank_lightgbm import LightGBMReranker

        return LightGBMReranker()
    from recolens.core.rank import LogisticReranker

    return LogisticReranker()


def _make_ranker(method: str, dim: int, reranker: str = "logistic"):
    from recolens.packs.ugc.baselines import ContentRanker, PopularityRanker
    from recolens.packs.ugc.reco_collab import CollaborativeRanker
    from recolens.packs.ugc.reco_hybrid import HybridRanker

    if method == "popularity":
        return PopularityRanker()
    if method == "content":
        return ContentRanker(dim=dim)
    if method == "collab":
        return CollaborativeRanker()
    if method == "hybrid":
        return HybridRanker(dim=dim)
    if method == "reranked":
        from recolens.packs.ugc.reco_reranked import RerankedRanker

        return RerankedRanker(dim=dim, model=_make_reranker_model(reranker))
    raise ValueError(f"unknown method: {method}")


def _register_ab(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--a", choices=_METHODS, default="popularity", help="variant A")
    sp.add_argument("--b", choices=_METHODS, default="hybrid", help="variant B")
    sp.add_argument("--k", type=int, default=10, help="KPI cutoff (hit-rate@k)")
    sp.add_argument("--n-items", type=int, default=300)
    sp.add_argument("--n-users", type=int, default=80)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--dim", type=int, default=64)
    sp.add_argument("--test-ratio", type=float, default=0.3)
    sp.add_argument("--out", default=None, help="run dir (default: runs/<ts>)")
    sp.set_defaults(_handler=_ab)


def _ab(args: argparse.Namespace) -> int:
    from recolens.core import manifest
    from recolens.core.ab import ab_compare
    from recolens.core.evaluate import build_qrels, time_split
    from recolens.core.schema import parse_interactions, parse_items
    from recolens.packs.ugc.synth import generate

    data = generate(n_items=args.n_items, n_users=args.n_users, seed=args.seed)
    items = parse_items(data["items"]).valid
    interactions = parse_interactions(data["interactions"]).valid
    train, test = time_split(interactions, test_ratio=args.test_ratio)
    qrels = build_qrels(test)
    users = sorted(qrels.keys())

    ra = _make_ranker(args.a, args.dim)
    rb = _make_ranker(args.b, args.dim)
    ra.fit(items, train)
    rb.fit(items, train)
    runs_a = {u: ra.rank(u, args.k) for u in users}
    runs_b = {u: rb.rank(u, args.k) for u in users}

    res = ab_compare(qrels, runs_a, runs_b, k=args.k, seed=args.seed)
    print(f"A = {args.a}   B = {args.b}   KPI = hit-rate@{args.k}   n={res['n']}")
    print(f"  KPI_A = {res['kpi_a']:.4f}")
    print(f"  KPI_B = {res['kpi_b']:.4f}")
    print(f"  abs lift = {res['abs_lift']:+.4f}   rel lift = {res['rel_lift']:+.2%}")
    print(f"  95% CI (B-A) = [{res['ci95_low']:+.4f}, {res['ci95_high']:+.4f}]")
    print(f"  decision: {res['decision']}")

    run_dir = manifest.new_run_dir() if args.out is None else args.out
    path = manifest.write_manifest(
        run_dir,
        {"stage": "ab", "variant_a": args.a, "variant_b": args.b, "seed": args.seed, "result": res},
    )
    print(f"\nmanifest: {path}")
    return 0


def _print_eval_table(results: dict, metric_names: list) -> None:
    col = max(12, max((len(n) for n in results), default=12))
    header = "metric".ljust(14) + "".join(n.rjust(col) for n in results)
    print(header)
    print("-" * len(header))
    for m in metric_names:
        row = m.ljust(14) + "".join(f"{results[n].get(m, 0.0):.4f}".rjust(col) for n in results)
        print(row)


def _register_recommend(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--user", default=None, help="user id (default: first user)")
    sp.add_argument("--method", choices=_METHODS, default="hybrid")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--n-items", type=int, default=300)
    sp.add_argument("--n-users", type=int, default=80)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--dim", type=int, default=64)
    sp.set_defaults(_handler=_recommend)


def _recommend(args: argparse.Namespace) -> int:
    from recolens.core.schema import parse_interactions, parse_items
    from recolens.packs.ugc.synth import generate

    data = generate(n_items=args.n_items, n_users=args.n_users, seed=args.seed)
    items = parse_items(data["items"]).valid
    interactions = parse_interactions(data["interactions"]).valid
    user = args.user or data["users"][0]["user_id"]

    ranker = _make_ranker(args.method, args.dim)
    ranker.fit(items, interactions)
    recs = ranker.rank(user, args.k)
    by_id = {it.item_id: it for it in items}

    print(f"recommend for {user}  (method={args.method}, k={args.k})")
    for rank, iid in enumerate(recs, 1):
        it = by_id.get(iid)
        tags = ",".join(it.tags) if it else ""
        title = it.title if it else ""
        print(f"{rank:2}. {iid}  [{tags}]  {title}")
    if not recs:
        print("  (no recommendations)")
    return 0


def _make_llm_provider(name: str):
    if name == "ollama":
        from recolens.providers.llm_ollama import OllamaLLM

        return OllamaLLM()
    from recolens.providers.llm_mock import DeterministicMockLLM

    return DeterministicMockLLM()


def _register_classify(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("text", help="content text to classify")
    sp.add_argument("--llm", choices=["mock", "ollama"], default="mock")
    sp.set_defaults(_handler=_classify)


def _classify(args: argparse.Namespace) -> int:
    from recolens.packs.ugc.classify import classify

    provider = _make_llm_provider(args.llm)
    r = classify(args.text, provider=provider)
    print(f"category : {r.category}  (confidence {r.confidence}, source {r.source})")
    print(f"quality  : {r.quality}")
    return 0


def _register_moderate(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("text", help="content text to moderate")
    sp.add_argument("--llm", choices=["mock", "ollama"], default="mock")
    sp.set_defaults(_handler=_moderate)


def _moderate(args: argparse.Namespace) -> int:
    from recolens.packs.ugc.moderate import moderate

    provider = _make_llm_provider(args.llm)
    r = moderate(args.text, provider=provider)
    print(f"action   : {r.action}  (malicious={r.malicious})")
    print(f"injection: {r.injection}   spam: {r.spam}   source: {r.source}")
    if r.reasons:
        print("reasons  : " + "; ".join(r.reasons))
    return 0


def _register_bench(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--n-items", type=int, default=300)
    sp.add_argument("--n-queries", type=int, default=50)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--dim", type=int, default=64, help="hash embedder dim")
    sp.add_argument("--all", action="store_true", help="include local-embed / qdrant if available")
    sp.add_argument("--out", default="docs/BENCHMARK.generated.md")
    sp.set_defaults(_handler=_bench)


def _bench(args: argparse.Namespace) -> int:
    from pathlib import Path

    from recolens.core.bench import bench_config
    from recolens.core.embedding import DeterministicHashEmbed
    from recolens.core.schema import parse_items
    from recolens.core.vector_index import InMemoryFlatIndex
    from recolens.packs.ugc.synth import generate

    items = parse_items(generate(n_items=args.n_items, n_users=1, seed=args.seed)["items"]).valid

    def mem_factory(_dim):
        return InMemoryFlatIndex()

    def qdrant_factory(dim):
        from recolens.providers.vector_qdrant import QdrantIndex

        return QdrantIndex(dim)

    # (label, embedder, index_factory) — built lazily so missing extras just skip.
    configs: list = [("hash + memory", DeterministicHashEmbed(dim=args.dim), mem_factory)]
    if args.all:
        try:
            from recolens.providers.embed_local import LocalSentenceTransformerEmbed

            local = LocalSentenceTransformerEmbed()
            configs.append(("e5-small + memory", local, mem_factory))
            configs.append(("e5-small + qdrant", local, qdrant_factory))
        except Exception as e:
            sys.stderr.write(f"local embed unavailable ({e}); skipping e5 configs\n")
        configs.append(("hash + qdrant", DeterministicHashEmbed(dim=args.dim), qdrant_factory))

    rows = []
    for label, embedder, factory in configs:
        try:
            r = bench_config(items, embedder, factory, n_queries=args.n_queries, k=args.k)
            rows.append((label, r))
            print(
                f"{label:22} dim={r['dim']:>4} embed={r['embed_ms_per_text']}ms/txt "
                f"build={r['build_ms']}ms p50={r['search_p50_ms']}ms p95={r['search_p95_ms']}ms "
                f"tag_match@{args.k}={r['tag_match_at_k']} mem~{r['index_mem_kb_est']}KB"
            )
        except Exception as e:
            sys.stderr.write(f"config '{label}' failed: {e}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_benchmark_md(rows, args), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def _render_benchmark_md(rows: list, args: argparse.Namespace) -> str:
    header = (
        "# Benchmark (generated)\n\n"
        f"Corpus = {args.n_items} items, {args.n_queries} queries, k={args.k}, CPU.\n"
        "Latency is wall-clock (varies run to run); regenerate with `recolens bench --all`.\n\n"
        "| config | dim | embed ms/text | build ms | search p50 ms | search p95 ms "
        f"| tag-match@{args.k} | index mem (KB est) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for label, r in rows:
        body += (
            f"| {label} | {r['dim']} | {r['embed_ms_per_text']} | {r['build_ms']} "
            f"| {r['search_p50_ms']} | {r['search_p95_ms']} | {r['tag_match_at_k']} "
            f"| {r['index_mem_kb_est']} |\n"
        )
    note = (
        "\n**Reading it**: the hash embedder is near-zero cost and dependency-free "
        "(good default / fast iteration); e5-small trades higher embed cost for better "
        "semantic quality (see docs/evidence/embedding_spike_b1.md). Qdrant adds a small "
        "indexing overhead but scales beyond in-memory. Pick per the quality/latency/cost budget.\n"
    )
    return header + body + note


def _not_implemented(name: str):
    def _handler(_args: argparse.Namespace) -> int:
        sys.stderr.write(f"recolens {name}: not implemented yet\n")
        return 2

    return _handler


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args._handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
