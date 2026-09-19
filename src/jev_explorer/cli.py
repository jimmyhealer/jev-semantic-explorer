from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jev_explorer.config import repo_root
from jev_explorer.index import build_index, dump_index
from jev_explorer.investigate import codebase_investigate
from jev_explorer.models import Budget, EvidencePacket, EvidencePolicy, InvestigateRequest, Scope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jevex",
        description="Stop grepping. Ask a repo where behavior is actually enforced.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_p = sub.add_parser("index", help="Build a reusable repo index.")
    index_p.add_argument("repo", nargs="?", default=None)
    index_p.add_argument("-o", "--output", default=None)

    inv_p = sub.add_parser("investigate", help="Ask an engineering question and return an evidence packet.")
    inv_p.add_argument("question")
    inv_p.add_argument("--repo", default=None)
    inv_p.add_argument("--path", action="append", dest="paths", default=[])
    inv_p.add_argument("--symbol", action="append", dest="symbols", default=[])
    inv_p.add_argument("--max-nodes", type=int, default=18)
    inv_p.add_argument("--max-depth", type=int, default=3)
    inv_p.add_argument("--max-branch", type=int, default=4)
    inv_p.add_argument("--max-jev-tokens", type=int, default=80_000)
    inv_p.add_argument("--index", dest="index_path", default=None, help="Reuse a dumped .jev-index.json")
    inv_p.add_argument("--no-jev", action="store_true", help="BM25+graph packet without Jev (ablation).")
    inv_p.add_argument("--require-tests", action="store_true")
    inv_p.add_argument("--require-docs", action="store_true")
    inv_p.add_argument("--allow-generated", action="store_true")
    inv_p.add_argument("--json", action="store_true", help="Print the full evidence packet as JSON.")

    eval_p = sub.add_parser("eval", help="Run the MVP eval harness against a fixture repo.")
    eval_p.add_argument("--repo", default=None)
    eval_p.add_argument("--cases", default=None)
    eval_p.add_argument("--max-nodes", type=int, default=16)

    bench_p = sub.add_parser("bench", help="Run public-bench harnesses (swe / loc / arb).")
    bench_p.add_argument("suite", choices=["swe", "loc", "arb"])
    bench_p.add_argument("--bench", default=None)
    bench_p.add_argument("--arms", default="bm25,explorer")
    bench_p.add_argument("--limit", type=int, default=0)
    bench_p.add_argument("--use-jev", action="store_true")
    bench_p.add_argument("--force-claude", action="store_true")
    bench_p.add_argument("--output", default=None)

    args = parser.parse_args(argv)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "investigate":
        return cmd_investigate(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "bench":
        return cmd_bench(args)
    parser.error("unknown command")
    return 2


def default_repo(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    fixture = repo_root() / "fixtures" / "sample_app"
    return fixture if fixture.exists() else Path.cwd()


def cmd_index(args: argparse.Namespace) -> int:
    repo = default_repo(args.repo)
    index = build_index(repo)
    output = Path(args.output) if args.output else repo / ".jev-index.json"
    dump_index(index, output)
    print(
        json.dumps(
            {
                "root": index.root,
                "artifacts": len(index.artifacts),
                "edges": len(index.edges),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


def cmd_investigate(args: argparse.Namespace) -> int:
    repo = default_repo(args.repo)
    request = InvestigateRequest(
        question=args.question,
        repo=str(repo),
        scope=Scope(paths=args.paths, symbols=args.symbols),
        budget=Budget(
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            max_branch=args.max_branch,
            max_jev_tokens=args.max_jev_tokens,
        ),
        evidence_policy=EvidencePolicy(
            require_tests=args.require_tests,
            require_docs=args.require_docs,
            allow_generated=args.allow_generated,
        ),
        use_jev=not args.no_jev,
        index_path=args.index_path,
    )
    packet = codebase_investigate(request)
    if args.json:
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        print(format_packet_text(packet))
    return 0


def format_packet_text(packet: EvidencePacket) -> str:
    metrics = packet.metrics or {}
    latency = metrics.get("latency_ms")
    files = metrics.get("file_count")
    header = f"status  {packet.status}"
    extras = []
    if latency is not None:
        extras.append(f"{round(float(latency))}ms")
    if files is not None:
        extras.append(f"{files} files")
    if extras:
        header += "  ·  " + " · ".join(str(item) for item in extras)
    lines = [header, "", "source of truth"]
    if not packet.source_of_truth:
        lines.append("  (none)")
    for item in packet.source_of_truth[:5]:
        span = f":{item.range[0]}-{item.range[1]}" if item.range else ""
        symbol = f"  {item.symbol}" if item.symbol else ""
        lines.append(f"  {item.path}{span}{symbol}")
    lines.append("")
    lines.append("read next")
    if not packet.regions:
        lines.append("  (none)")
    for region in packet.regions[:5]:
        path = region.get("path") or region.get("file")
        start = region.get("start") or region.get("start_line")
        end = region.get("end") or region.get("end_line")
        symbol = region.get("symbol") or ""
        suffix = f"  {symbol}" if symbol else ""
        lines.append(f"  {path}:{start}-{end}{suffix}")
    supporting = packet.supporting
    if supporting.tests or supporting.callers:
        lines.append("")
        if supporting.tests:
            lines.append("tests     " + ", ".join(supporting.tests[:4]))
        if supporting.callers:
            lines.append("callers   " + ", ".join(supporting.callers[:4]))
    if packet.unknowns:
        lines.append("")
        lines.append("unknowns")
        for item in packet.unknowns[:4]:
            lines.append(f"  {item}")
    return "\n".join(lines)


def cmd_eval(args: argparse.Namespace) -> int:
    from jev_explorer.eval_harness import run_eval

    repo = default_repo(args.repo)
    cases = Path(args.cases) if args.cases else repo_root() / "eval" / "cases.json"
    result = run_eval(repo=repo, cases_path=cases, max_nodes=args.max_nodes)
    print(json.dumps(result, indent=2))
    failed = result["summary"]["failed"]
    return 1 if failed else 0


def cmd_bench(args: argparse.Namespace) -> int:
    from importlib.machinery import SourceFileLoader

    from jev_explorer.config import repo_root as root

    if args.suite == "swe":
        sys.argv = [
            "swe-explore",
            "--bench",
            args.bench or str(root() / "eval" / "swe_explore" / "fixture_bench.jsonl"),
            "--arms",
            args.arms,
            "--output",
            args.output or str(root() / "eval" / "runs" / "swe_explore.json"),
        ]
        if args.limit:
            sys.argv += ["--limit", str(args.limit)]
        if args.force_claude:
            sys.argv.append("--force-claude")
        module = SourceFileLoader("swe_run", str(root() / "eval" / "swe_explore" / "run.py")).load_module()
        return module.main()
    if args.suite == "loc":
        sys.argv = [
            "loc-bench",
            "--bench",
            args.bench or str(root() / "eval" / "loc_bench" / "fixture.json"),
            "--output",
            args.output or str(root() / "eval" / "runs" / "loc_bench.json"),
        ]
        if args.use_jev:
            sys.argv.append("--use-jev")
        if args.limit:
            sys.argv += ["--limit", str(args.limit)]
        module = SourceFileLoader("loc_run", str(root() / "eval" / "loc_bench" / "run.py")).load_module()
        return module.main()
    sys.argv = [
        "arb",
        "--bench",
        args.bench or str(root() / "eval" / "arb" / "fixture.json"),
        "--output",
        args.output or str(root() / "eval" / "runs" / "arb.json"),
    ]
    if args.use_jev:
        sys.argv.append("--use-jev")
    if args.limit:
        sys.argv += ["--limit", str(args.limit)]
    module = SourceFileLoader("arb_run", str(root() / "eval" / "arb" / "run.py")).load_module()
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
