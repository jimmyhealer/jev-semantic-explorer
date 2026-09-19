from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jev_explorer.config import load_skill_text, repo_root
from jev_explorer.coding_agent import agent_model, invoke_coding_agent
from jev_explorer.index import get_or_build_index
from jev_explorer.investigate import codebase_investigate
from jev_explorer.models import Budget, EvidencePacket, InvestigateRequest, Scope
from jev_explorer.swe_metrics import mean_metrics, score_prediction

AGENT_TIMEOUT_SEC = 1200

PAPER_CLAUDE_HITFILE = 0.667
# Vendor list prices for estimates, not invoices.
# Jev: TypeSafe bills input only; output is free (too cheap to meter).
JEV_INPUT_USD_PER_MTOK = 0.042
JEV_OUTPUT_USD_PER_MTOK = 0.0
# Gemini 3.8 Flash introductory rates through 2026-12-31. Output includes thinking.
GEMINI_FLASH_INPUT_USD_PER_MTOK = 0.75
GEMINI_FLASH_OUTPUT_USD_PER_MTOK = 3.75
GEMINI_FLASH_CACHE_USD_PER_MTOK = 0.075
PRICING = {
    "as_of": "2026-09-19",
    "jev_input_usd_per_mtok": JEV_INPUT_USD_PER_MTOK,
    "jev_output_usd_per_mtok": JEV_OUTPUT_USD_PER_MTOK,
    "gemini_3_8_flash_input_usd_per_mtok": GEMINI_FLASH_INPUT_USD_PER_MTOK,
    "gemini_3_8_flash_output_usd_per_mtok": GEMINI_FLASH_OUTPUT_USD_PER_MTOK,
    "gemini_3_8_flash_cache_usd_per_mtok": GEMINI_FLASH_CACHE_USD_PER_MTOK,
    "notes": [
        "Jev USD = billed input tokens × $0.042 / MTok; output is $0.",
        "Gemini 3.8 Flash USD uses introductory list prices; thinking tokens billed as output.",
        "Estimates, not provider invoices. Cache writes are not separately metered here.",
    ],
}
REGIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["file", "start_line", "end_line"],
            },
        }
    },
    "required": ["regions"],
}


def load_instances(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict) and "instances" in payload:
            return payload["instances"]
        if isinstance(payload, list):
            return payload
        return [payload]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def resolve_repo(instance: dict[str, Any], repos_root: Path | None = None) -> Path:
    if instance.get("dataset") == "fixture" or str(instance.get("instance_id", "")).startswith("fixture__"):
        return repo_root() / instance.get("repo_dir", "fixtures/sample_app")
    if repos_root:
        repo_dir = instance.get("repo_dir") or instance["instance_id"]
        candidate = repos_root / Path(repo_dir).name
        if candidate.exists():
            return candidate
        nested = repos_root / repo_dir
        if nested.exists():
            return nested
    explicit = instance.get("repo_path")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    raise FileNotFoundError(f"No local snapshot for {instance['instance_id']}")


def ensure_index(repo: Path) -> str | None:
    get_or_build_index(repo)
    return None


def packet_to_preds(packet: EvidencePacket) -> list[tuple[str, int, int]]:
    preds: list[tuple[str, int, int]] = []
    for region in packet.regions:
        preds.append((region["path"], int(region["start"]), int(region["end"])))
    if preds:
        return preds
    for item in packet.source_of_truth:
        preds.append((item.path, int(item.range[0]), int(item.range[1])))
    if preds:
        return preds
    for path in packet.metrics.get("files_touched") or []:
        preds.append((str(path), 1, 40))
    return preds[:5]


def run_explorer_arm(instance: dict[str, Any], repo: Path, use_jev: bool, max_nodes: int = 18) -> dict[str, Any]:
    started = time.perf_counter()
    packet = codebase_investigate(
        InvestigateRequest(
            question=instance["problem_statement"] or instance["instance_id"],
            repo=str(repo),
            scope=Scope(),
            budget=Budget(max_nodes=max_nodes, max_depth=3, max_branch=4),
            use_jev=use_jev,
            index_path=ensure_index(repo),
        )
    )
    preds = packet_to_preds(packet)
    metrics = score_prediction(preds, instance["ground_truth"])
    metrics.update(
        {
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "jev_calls": packet.metrics.get("jev_calls", 0),
            "jev_tokens": packet.metrics.get("jev_tokens", 0),
            "jev_input_tokens": packet.metrics.get("jev_input_tokens", 0),
            "jev_output_tokens": packet.metrics.get("jev_output_tokens", 0),
            "file_count": packet.metrics.get("file_count", len(preds)),
            "status": packet.status,
        }
    )
    return {
        "instance_id": instance["instance_id"],
        "arm": "explorer" if use_jev else "bm25",
        "preds": [{"file": p, "start_line": s, "end_line": e} for p, s, e in preds],
        "metrics": metrics,
        "unknowns": packet.unknowns,
        "packet_status": packet.status,
    }


def run_agent_arm(
    instance: dict[str, Any],
    repo: Path,
    packet: dict[str, Any] | None = None,
    timeout: int = AGENT_TIMEOUT_SEC,
) -> dict[str, Any]:
    started = time.perf_counter()
    skill = load_skill_text()
    issue = instance.get("problem_statement") or instance["instance_id"]
    if packet is None:
        prompt = (
            f"{skill}\n\n"
            "The codebase_investigate tool is not available in this arm. "
            "Use Read/Grep/Glob only. Do not edit files.\n"
            "Return JSON only: {\"regions\": [{\"file\": \"path\", \"start_line\": 1, \"end_line\": 20}]}\n"
            f"Issue:\n{issue}\n"
            "Rank at most 5 regions that a patching agent should read. Never return an empty regions list."
        )
        arm = "agy"
    else:
        prompt = (
            f"{skill}\n\n"
            "An evidence packet from codebase_investigate is already provided. "
            "Treat it as the tool result. Read cited files only. Do not Grep the rest of the repo "
            "unless status is unknown or insufficient_evidence.\n"
            "Do not edit files. Return JSON regions as specified in the skill. Never return an empty list.\n"
            f"Issue:\n{issue}\n"
            f"Evidence packet:\n{json.dumps(packet)[:12000]}\n"
        )
        arm = "agy+explorer"
    raw = invoke_coding_agent(repo, prompt, json_schema=REGIONS_SCHEMA, timeout=timeout)
    structured = raw.get("structured_output") or {}
    regions = raw.get("regions") or structured.get("regions") or []
    preds = [
        (
            item.get("file") or item.get("path"),
            int(item.get("start_line") or item.get("start") or 1),
            int(item.get("end_line") or item.get("end") or 1),
        )
        for item in regions
        if item.get("file") or item.get("path")
    ]
    empty = (not preds) or bool(raw.get("error"))
    metrics = score_prediction(preds, instance["ground_truth"])
    usage = raw.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    thinking_tokens = int(usage.get("thinking_tokens") or 0)
    cache_read_tokens = int(usage.get("cache_read_tokens") or 0)
    metrics.update(
        {
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "tool_calls": raw.get("_tool_calls") or 0,
            "agent_model": raw.get("model") or agent_model(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "cache_read_tokens": cache_read_tokens,
            "estimated_usd": round(
                estimate_gemini_usd(input_tokens, output_tokens, thinking_tokens, cache_read_tokens),
                6,
            ),
            "empty_output": empty,
            "file_count": len({p[0] for p in preds}),
        }
    )
    return {
        "instance_id": instance["instance_id"],
        "arm": arm,
        "agent": raw.get("agent"),
        "model": raw.get("model") or agent_model(),
        "preds": [{"file": p, "start_line": s, "end_line": e} for p, s, e in preds],
        "metrics": metrics,
        "error": raw.get("error"),
        "raw": {k: v for k, v in raw.items() if k not in {"_tool_calls", "raw"}},
    }


def run_claude_arm(instance: dict[str, Any], repo: Path, packet: dict[str, Any] | None = None) -> dict[str, Any]:
    """Back-compat alias; coding-agent arms are agy."""
    row = run_agent_arm(instance, repo, packet=packet)
    row["arm"] = "claude+mcp" if packet is not None else "claude"
    return row


def estimate_jev_usd(input_tokens: float, output_tokens: float = 0.0) -> float:
    return (input_tokens * JEV_INPUT_USD_PER_MTOK + output_tokens * JEV_OUTPUT_USD_PER_MTOK) / 1_000_000


def estimate_gemini_usd(
    input_tokens: float,
    output_tokens: float = 0.0,
    thinking_tokens: float = 0.0,
    cache_read_tokens: float = 0.0,
) -> float:
    billed_output = output_tokens + thinking_tokens
    return (
        input_tokens * GEMINI_FLASH_INPUT_USD_PER_MTOK
        + billed_output * GEMINI_FLASH_OUTPUT_USD_PER_MTOK
        + cache_read_tokens * GEMINI_FLASH_CACHE_USD_PER_MTOK
    ) / 1_000_000


def spend_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    jev_calls = sum(int(row.get("metrics", {}).get("jev_calls") or 0) for row in rows)
    jev_input = sum(int(row.get("metrics", {}).get("jev_input_tokens") or 0) for row in rows)
    jev_output = sum(int(row.get("metrics", {}).get("jev_output_tokens") or 0) for row in rows)
    jev_total = sum(int(row.get("metrics", {}).get("jev_tokens") or 0) for row in rows)
    billed_input = jev_input if jev_input else jev_total
    gemini_in = sum(int(row.get("metrics", {}).get("input_tokens") or 0) for row in rows)
    gemini_out = sum(int(row.get("metrics", {}).get("output_tokens") or 0) for row in rows)
    gemini_think = sum(int(row.get("metrics", {}).get("thinking_tokens") or 0) for row in rows)
    gemini_cache = sum(int(row.get("metrics", {}).get("cache_read_tokens") or 0) for row in rows)
    jev_usd = estimate_jev_usd(billed_input, jev_output)
    gemini_usd = estimate_gemini_usd(gemini_in, gemini_out, gemini_think, gemini_cache)
    return {
        "n": len(rows),
        "jev_calls": jev_calls,
        "jev_input_tokens": billed_input,
        "jev_output_tokens": jev_output,
        "jev_tokens": jev_total or (billed_input + jev_output),
        "jev_estimated_usd": round(jev_usd, 6),
        "gemini_input_tokens": gemini_in,
        "gemini_output_tokens": gemini_out,
        "gemini_thinking_tokens": gemini_think,
        "gemini_cache_read_tokens": gemini_cache,
        "gemini_estimated_usd": round(gemini_usd, 6),
        "estimated_usd": round(jev_usd + gemini_usd, 6),
        "pricing": PRICING,
    }


def summarize_spend(arm_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_arm = {arm: spend_from_rows(rows) for arm, rows in arm_rows.items() if rows}
    totals = spend_from_rows([row for rows in arm_rows.values() for row in rows])
    return {"by_arm": by_arm, "total": totals, "pricing": PRICING}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row["metrics"] for row in rows if not row.get("metrics", {}).get("empty_output")]
    summary = mean_metrics(scored) if scored else {}
    n = len(rows)
    n_scored = len(scored)
    n_empty = sum(1 for row in rows if row.get("metrics", {}).get("empty_output"))
    summary["n"] = n
    summary["n_scored"] = n_scored
    summary["n_empty"] = n_empty
    summary["completion_rate"] = round(n_scored / n, 4) if n else 0.0
    hit_scored = float(summary.get("hit_file_rate") or 0)
    summary["hit_file_rate_all"] = round(hit_scored * n_scored / n, 4) if n else 0.0
    summary.update({f"spend_{key}": value for key, value in spend_from_rows(rows).items() if key != "pricing"})
    return summary


def agent_gates(arm_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = arm_summaries.get("agy") or arm_summaries.get("claude") or {}
    explorer = arm_summaries.get("agy+explorer") or arm_summaries.get("claude+mcp") or {}
    notes = [
        "Product test is agy vs agy+explorer on the same coding agent.",
        "Empty structured output is a failed row, excluded from HitFile means.",
        "Packet-only SWE-Explore HitFile is diagnostic, not a launch gate.",
    ]
    if not baseline or not explorer:
        return {
            "recall_hold": False,
            "efficiency_win": False,
            "empty_output_ok": False,
            "pass_product": False,
            "incomplete": True,
            "notes": notes + ["Need both agy and agy+explorer summaries."],
        }
    hit_b = float(baseline.get("hit_file_rate") or 0)
    hit_e = float(explorer.get("hit_file_rate") or 0)
    files_b = float(baseline.get("file_count") or 0)
    files_e = float(explorer.get("file_count") or 0)
    tools_b = float(baseline.get("tool_calls") or 0)
    tools_e = float(explorer.get("tool_calls") or 0)
    tokens_b = float(baseline.get("input_tokens") or 0)
    tokens_e = float(explorer.get("input_tokens") or 0)
    lat_b = float(baseline.get("latency_ms") or 0)
    lat_e = float(explorer.get("latency_ms") or 0)
    recall_hold = hit_e + 0.01 >= hit_b
    efficiency_win = (
        (files_e and files_b and files_e < files_b)
        or (tools_e and tools_b and tools_e < tools_b)
        or (tokens_e and tokens_b and tokens_e < tokens_b)
        or (lat_e and lat_b and lat_e < lat_b)
    )
    empty_ok = int(baseline.get("n_empty") or 0) == 0 and int(explorer.get("n_empty") or 0) == 0
    scored = int(baseline.get("n_scored") or 0) >= 1 and int(explorer.get("n_scored") or 0) >= 1
    return {
        "recall_hold": recall_hold,
        "efficiency_win": efficiency_win,
        "empty_output_ok": empty_ok,
        "pass_product": bool(recall_hold and efficiency_win and empty_ok and scored),
        "hit_file_agy": hit_b,
        "hit_file_agy_explorer": hit_e,
        "hit_file_agy_all": float(baseline.get("hit_file_rate_all") or 0),
        "hit_file_agy_explorer_all": float(explorer.get("hit_file_rate_all") or 0),
        "completion_agy": float(baseline.get("completion_rate") or 0),
        "completion_agy_explorer": float(explorer.get("completion_rate") or 0),
        "notes": notes,
    }


def gates(arm_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if arm_summaries.get("agy") or arm_summaries.get("agy+explorer"):
        return agent_gates(arm_summaries)
    bm25 = arm_summaries.get("bm25", {})
    explorer = arm_summaries.get("explorer", {})
    notes = [
        "Packet vs BM25 is a diagnostic, not the product test.",
        "Do not skip agy arms because packet HitFile is below 0.55.",
        "Do not launch on fixture or packet-only scores.",
    ]
    if not bm25 or not explorer:
        return {
            "beat_bm25": False,
            "near_paper_claude_hitfile": False,
            "pass_to_claude_arms": True,
            "pass_to_phase4": False,
            "incomplete": True,
            "paper_claude_hitfile": PAPER_CLAUDE_HITFILE,
            "notes": notes + ["Diagnostic gate needs bm25 and explorer summaries."],
        }
    hit_bm25 = bm25.get("hit_file_rate", 0)
    hit_explorer = explorer.get("hit_file_rate", 0)
    ctx_bm25 = bm25.get("context_efficiency", 0)
    ctx_explorer = explorer.get("context_efficiency", 0)
    tied = abs(hit_explorer - hit_bm25) <= 0.01
    beat_bm25 = hit_explorer > hit_bm25 + 0.01 or (tied and ctx_explorer >= ctx_bm25)
    near_claude = hit_explorer + 0.15 >= PAPER_CLAUDE_HITFILE
    return {
        "beat_bm25": beat_bm25,
        "near_paper_claude_hitfile": near_claude,
        "pass_to_claude_arms": True,
        "pass_to_phase4": False,
        "paper_claude_hitfile": PAPER_CLAUDE_HITFILE,
        "notes": notes,
    }


def write_honesty(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
