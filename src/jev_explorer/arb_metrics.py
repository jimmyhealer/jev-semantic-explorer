from __future__ import annotations

from typing import Any


def mrr(ranked: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for i, path in enumerate(ranked, start=1):
        if path in gold_set:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 1.0
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


def should_abstain(row: dict[str, Any]) -> bool:
    subset = str(row.get("subset") or row.get("track") or row.get("task_type") or "").lower()
    if "abstain" in subset or "no-gold" in subset or "nogo" in subset or "abstention" in subset:
        return True
    gold = row.get("gold_files") or row.get("gold") or []
    if isinstance(gold, dict):
        return bool(gold.get("no_gold")) or not (gold.get("root_cause_files") or gold.get("files"))
    return not gold


def score_arb(ranked: list[str], status: str, row: dict[str, Any]) -> dict[str, Any]:
    gold = list(row.get("gold_files") or row.get("gold") or [])
    abstain_gold = should_abstain(row)
    abstained = status in {"unknown", "insufficient_evidence"}
    if abstain_gold:
        return {
            "selective_success": 1.0 if abstained else 0.0,
            "mrr": None,
            "recall@5": None,
            "recall@20": None,
        }
    return {
        "selective_success": None,
        "mrr": round(mrr(ranked, gold), 4),
        "recall@5": round(recall_at_k(ranked, gold, 5), 4),
        "recall@20": round(recall_at_k(ranked, gold, 20), 4),
    }
