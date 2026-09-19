from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from jev_explorer.index import build_index, iter_source_files, read_text
from jev_explorer.investigate import codebase_investigate
from jev_explorer.models import Budget, EvidencePolicy, InvestigateRequest, Scope
from jev_explorer.retrieval import STOPWORDS, tokenize

STOPWORDS_EXTRA = STOPWORDS | {"where", "which", "would", "changing", "function"}


def run_eval(repo: Path, cases_path: Path, max_nodes: int = 16) -> dict[str, Any]:
    cases = load_cases(cases_path)
    index = build_index(repo)
    rows: list[dict[str, Any]] = []
    for case in cases:
        explorer = run_explorer(repo, case, max_nodes)
        baseline = run_baseline(repo, case)
        rows.append(compare_case(case, explorer, baseline, index_size=len(index.artifacts)))

    passed = sum(1 for row in rows if row["pass"])
    failed = len(rows) - passed
    explorer_files = mean(row["explorer"]["file_count"] for row in rows)
    baseline_files = mean(row["baseline"]["file_count"] for row in rows)
    explorer_tokens = mean(row["explorer"]["agent_tokens"] for row in rows)
    baseline_tokens = mean(row["baseline"]["tokens"] for row in rows)
    explorer_latency = mean(row["explorer"]["latency_ms"] for row in rows)
    baseline_latency = mean(row["baseline"]["latency_ms"] for row in rows)
    explorer_tools = mean(row["explorer"]["agent_tool_calls"] for row in rows)
    baseline_tools = mean(row["baseline"]["tool_calls"] for row in rows)
    recall_ok = all(row["explorer"]["primary_source_recall"] >= row["baseline"]["primary_source_recall"] for row in rows)

    return {
        "repo": str(repo),
        "cases": rows,
        "summary": {
            "passed": passed,
            "failed": failed,
            "primary_source_recall_held": recall_ok,
            "avg_files_explorer": round(explorer_files, 2),
            "avg_files_baseline": round(baseline_files, 2),
            "avg_agent_tokens_explorer": round(explorer_tokens, 1),
            "avg_agent_tokens_baseline": round(baseline_tokens, 1),
            "avg_agent_tool_calls_explorer": round(explorer_tools, 2),
            "avg_agent_tool_calls_baseline": round(baseline_tools, 2),
            "avg_latency_ms_explorer": round(explorer_latency, 1),
            "avg_latency_ms_baseline": round(baseline_latency, 1),
            "avg_jev_tokens": round(mean(row["explorer"]["jev_tokens"] for row in rows), 1),
            "file_reduction": pct_drop(baseline_files, explorer_files),
            "agent_token_reduction": pct_drop(baseline_tokens, explorer_tokens),
            "agent_tool_call_reduction": pct_drop(baseline_tools, explorer_tools),
            "notes": [
                "Baseline is lexical grep+read, not a coding agent, so latency is not comparable.",
                "Agent tokens are the evidence packet the model would read, not internal Jev tokens.",
                "Success requires primary-source recall to hold; token/file/tool-call reductions are secondary.",
            ],
        },
    }


def run_explorer(repo: Path, case: dict[str, Any], max_nodes: int) -> dict[str, Any]:
    request = InvestigateRequest(
        question=case["question"],
        repo=str(repo),
        scope=Scope(paths=case.get("scope_paths", [])),
        budget=Budget(max_nodes=max_nodes, max_depth=3, max_branch=4),
        evidence_policy=EvidencePolicy(require_tests=case.get("require_tests", False)),
    )
    packet = codebase_investigate(request)
    data = packet.to_dict()
    hits = packet_paths(data)
    recall = primary_recall(case["primary_sources"], hits, data)
    agent_view = {
        "question": data["question"],
        "source_of_truth": data["source_of_truth"],
        "supporting": data["supporting"],
        "impact_paths": data["impact_paths"],
        "unknowns": data["unknowns"],
        "alternative_paths": data["alternative_paths"],
        "status": data["status"],
    }
    return {
        "packet": data,
        "paths": sorted(hits),
        "file_count": len(hits),
        "internal_files_touched": data["metrics"].get("file_count", len(hits)),
        "agent_tokens": estimate_tokens(json.dumps(agent_view)),
        "jev_tokens": data["metrics"].get("jev_tokens", 0),
        "latency_ms": data["metrics"].get("latency_ms", 0),
        "tool_calls": data["metrics"].get("jev_calls", 0),
        "agent_tool_calls": 1,
        "primary_source_recall": recall,
        "unknowns": data.get("unknowns", []),
        "status": data.get("status"),
    }


def run_baseline(repo: Path, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    keywords = [token for token in tokenize(case["question"]) if token not in STOPWORDS_EXTRA]
    files = iter_source_files(Path(repo), case.get("scope_paths"))
    matched: list[tuple[str, int, str]] = []
    bytes_read = 0
    for path in files:
        rel = path.relative_to(repo).as_posix()
        text = read_text(path)
        hits = sum(len(re.findall(re.escape(keyword), text, flags=re.I)) for keyword in keywords)
        if hits:
            matched.append((rel, hits, text))
            bytes_read += len(text.encode("utf-8"))
    matched.sort(key=lambda item: item[1], reverse=True)
    guessed = [{"path": path, "symbol": None} for path, _hits, _text in matched]
    latency_ms = (time.perf_counter() - started) * 1000
    recall = primary_recall(case["primary_sources"], {path for path, _h, _t in matched}, {"source_of_truth": guessed})
    return {
        "paths": [path for path, _h, _t in matched],
        "file_count": len(matched),
        "tokens": estimate_tokens("".join(text for _p, _h, text in matched)),
        "latency_ms": round(latency_ms, 1),
        "tool_calls": len(matched) + 1,
        "primary_source_recall": recall,
        "bytes_read": bytes_read,
    }


def compare_case(
    case: dict[str, Any],
    explorer: dict[str, Any],
    baseline: dict[str, Any],
    index_size: int,
) -> dict[str, Any]:
    recall_ok = explorer["primary_source_recall"] >= 1.0
    files_ok = explorer["file_count"] <= max(baseline["file_count"], 1)
    honest = True
    expected_status = case.get("expect_status")
    if expected_status:
        honest = explorer["status"] in expected_status
    elif case.get("expect_unknown"):
        honest = bool(explorer["unknowns"]) or explorer["status"] in {
            "insufficient_evidence",
            "unknown",
            "conflict",
        }
    must_include = set(case.get("must_include_paths", []))
    include_ok = must_include.issubset(set(explorer["paths"]))
    passed = recall_ok and honest and include_ok
    return {
        "id": case["id"],
        "question": case["question"],
        "pass": passed,
        "primary_source_hit": recall_ok,
        "honesty_ok": honest,
        "include_ok": include_ok,
        "files_not_worse": files_ok,
        "index_size": index_size,
        "explorer": {
            "primary_source_recall": explorer["primary_source_recall"],
            "file_count": explorer["file_count"],
            "agent_tokens": explorer["agent_tokens"],
            "jev_tokens": explorer["jev_tokens"],
            "latency_ms": explorer["latency_ms"],
            "tool_calls": explorer["tool_calls"],
            "agent_tool_calls": explorer["agent_tool_calls"],
            "status": explorer["status"],
            "paths": explorer["paths"],
            "unknowns": explorer["unknowns"],
            "source_of_truth": explorer["packet"]["source_of_truth"],
            "supporting": explorer["packet"]["supporting"],
        },
        "baseline": {
            "primary_source_recall": baseline["primary_source_recall"],
            "file_count": baseline["file_count"],
            "tokens": baseline["tokens"],
            "latency_ms": baseline["latency_ms"],
            "tool_calls": baseline["tool_calls"],
            "paths": baseline["paths"],
        },
    }


def packet_paths(packet: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for item in packet.get("source_of_truth", []):
        paths.add(item["path"])
    for group in packet.get("supporting", {}).values():
        paths.update(group)
    return paths


def primary_recall(expected: list[dict[str, Any]], paths: set[str], packet: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    hits = 0
    sources = packet.get("source_of_truth", [])
    for item in expected:
        path_ok = item["path"] in paths or any(src.get("path") == item["path"] for src in sources)
        symbol = item.get("symbol")
        symbol_ok = True
        if symbol:
            symbol_ok = any(
                src.get("path") == item["path"] and src.get("symbol") == symbol for src in sources
            ) or any(
                src.get("path") == item["path"] and not src.get("symbol") for src in sources
            )
        if path_ok and (symbol_ok or item["path"] in paths):
            # Path match is enough for recall; symbol match is preferred but graph support may only keep path.
            hits += 1
    return hits / len(expected)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def mean(values: list[float] | Any) -> float:
    seq = list(values)
    if not seq:
        return 0.0
    return sum(seq) / len(seq)


def pct_drop(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return round((before - after) / before * 100, 1)


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))
