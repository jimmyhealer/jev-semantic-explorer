from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import Any

from jev_explorer.index import get_or_build_index
from jev_explorer.jev_client import GapJudgment, JevJudge
from jev_explorer.models import (
    Artifact,
    Budget,
    EvidencePacket,
    InvestigateRequest,
    Judgment,
    NextEdge,
    RepoIndex,
    Role,
)
from jev_explorer.packet import build_packet
from jev_explorer.regions import ranked_regions
from jev_explorer.retrieval import clean_issue_text, seed_candidates


def search_question(request: InvestigateRequest) -> str:
    return clean_issue_text(request.question)

EDGE_TYPE_BY_NEXT: dict[NextEdge, set[str]] = {
    "callees": {"calls"},
    "callers": {"calls"},
    "tests": {"tested_by"},
    "docs": {"documented_by"},
    "config": {"contains", "imports"},
    "stop": set(),
}


@dataclass(order=True)
class FrontierItem:
    priority: float
    artifact_id: str = field(compare=False)
    depth: int = field(compare=False)
    via: str = field(compare=False, default="seed")
    seed_score: float = field(compare=False, default=0.0)


@dataclass
class SearchState:
    judgments: dict[str, Judgment] = field(default_factory=dict)
    visited: set[str] = field(default_factory=set)
    files_touched: set[str] = field(default_factory=set)
    gap: GapJudgment | None = None
    stop_reason: str = ""
    seed_scores: dict[str, float] = field(default_factory=dict)


async def codebase_investigate_async(request: InvestigateRequest) -> EvidencePacket:
    started = time.perf_counter()
    index = get_or_build_index(request.repo, request.scope.paths, request.index_path)
    judge = JevJudge() if request.use_jev else None
    search = await run_search(index, request, judge)
    packet = build_packet(request, index, search.judgments, search.gap)
    regions = ranked_regions(index, search.judgments, seed_scores=search.seed_scores)
    packet.regions = [item.to_dict() for item in regions]
    packet.metrics.update(
        {
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "artifacts_indexed": len(index.artifacts),
            "nodes_evaluated": len(search.judgments),
            "files_touched": sorted(search.files_touched),
            "file_count": len(search.files_touched),
            "jev_calls": judge.calls if judge else 0,
            "jev_tokens": judge.tokens_used if judge else 0,
            "jev_input_tokens": judge.input_tokens if judge else 0,
            "jev_output_tokens": judge.output_tokens if judge else 0,
            "stop_reason": search.stop_reason,
            "use_jev": request.use_jev,
            "index_cached": bool(request.index_path) or True,
        }
    )
    return packet


def codebase_investigate(request: InvestigateRequest) -> EvidencePacket:
    return asyncio.run(codebase_investigate_async(request))


async def run_search(
    index: RepoIndex,
    request: InvestigateRequest,
    judge: JevJudge | None,
) -> SearchState:
    if not request.use_jev or judge is None:
        return run_lexical_search(index, request)
    return await run_jev_search(index, request, judge)


def seed_frontier(index: RepoIndex, request: InvestigateRequest) -> tuple[list[FrontierItem], set[str], dict[str, float]]:
    budget = request.budget
    seeds = seed_candidates(
        index,
        request.question,
        request.scope.symbols,
        limit=max(12, budget.max_branch * 3),
        path_prefixes=request.scope.paths or None,
    )
    seed_ids = {artifact_id for artifact_id, _score in seeds}
    scores = {artifact_id: score for artifact_id, score in seeds}
    frontier: list[FrontierItem] = []
    for artifact_id, score in seeds:
        heapq.heappush(
            frontier,
            FrontierItem(priority=-score, artifact_id=artifact_id, depth=0, via="seed", seed_score=score),
        )
    return frontier, seed_ids, scores


def run_lexical_search(index: RepoIndex, request: InvestigateRequest) -> SearchState:
    budget = request.budget
    frontier, seed_ids, scores = seed_frontier(index, request)
    state = SearchState(seed_scores=scores)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    while frontier and len(state.judgments) < budget.max_nodes:
        item = heapq.heappop(frontier)
        if item.artifact_id in state.visited or item.depth > budget.max_depth:
            continue
        artifact = index.artifacts.get(item.artifact_id)
        if artifact is None:
            continue
        if artifact.kind == "generated" and not request.evidence_policy.allow_generated:
            state.visited.add(item.artifact_id)
            continue
        state.visited.add(item.artifact_id)
        state.files_touched.add(artifact.path)
        rank = next((i for i, (aid, _s) in enumerate(ranked) if aid == artifact.id), len(ranked))
        judgment = lexical_judgment(artifact, item.seed_score or scores.get(artifact.id, 1.0), rank)
        state.judgments[artifact.id] = judgment
        seed_ids.discard(artifact.id)
        if item.depth >= 1:
            continue
        fake = Judgment(**{**judgment.__dict__, "next_edge": "callers" if rank < 3 else "tests"})
        for neighbor_id, depth, via, score in expand_neighbors(index, artifact, fake, item.depth, budget):
            if neighbor_id in state.visited:
                continue
            heapq.heappush(
                frontier,
                FrontierItem(priority=-score, artifact_id=neighbor_id, depth=depth, via=via, seed_score=score),
            )
    promote_lexical_sot(index, state)
    state.stop_reason = "no_jev"
    if has_source_of_truth(state):
        state.stop_reason = "no_jev_sot"
    return state


def promote_lexical_sot(index: RepoIndex, state: SearchState) -> None:
    if has_source_of_truth(state):
        return
    ranked = []
    for artifact_id, judgment in state.judgments.items():
        artifact = index.artifacts.get(artifact_id)
        if artifact is None or artifact.kind not in {"function", "class"}:
            continue
        ranked.append((state.seed_scores.get(artifact_id, judgment.relevance), artifact_id, judgment))
    if not ranked:
        return
    ranked.sort(reverse=True)
    _score, artifact_id, judgment = ranked[0]
    state.judgments[artifact_id] = Judgment(
        **{
            **judgment.__dict__,
            "role": "source_of_truth",
            "role_confidence": max(judgment.role_confidence, 0.55),
            "relevance": max(judgment.relevance, 2.2),
            "is_source_of_truth": max(judgment.is_source_of_truth, 0.84),
        }
    )


def lexical_judgment(artifact: Artifact, score: float, rank: int) -> Judgment:
    role: Role
    if artifact.kind == "test":
        role = "test"
    elif artifact.kind == "docs":
        role = "docs"
    elif artifact.kind == "config":
        role = "config"
    elif artifact.kind == "generated":
        role = "generated"
    elif rank == 0 and artifact.kind in {"function", "class"}:
        role = "source_of_truth"
    elif artifact.kind in {"function", "class", "module"}:
        role = "implementation"
    else:
        role = "implementation"
    relevance = min(3.0, 1.0 + min(score, 8.0) / 4.0)
    if rank == 0 and artifact.kind in {"function", "class"}:
        relevance = max(relevance, 2.2)
        is_sot = 0.84
    elif rank < 3 and artifact.kind in {"function", "class"}:
        is_sot = 0.45
    else:
        is_sot = 0.12
    next_edge: NextEdge = "stop"
    if rank < 4:
        next_edge = "callers" if artifact.kind in {"function", "class"} else "tests"
    return Judgment(
        artifact_id=artifact.id,
        role=role,
        role_confidence=0.55 if role == "source_of_truth" else 0.4,
        role_probabilities={role: 1.0},
        relevance=round(relevance, 3),
        relevance_confidence=0.4,
        is_source_of_truth=is_sot,
        next_edge=next_edge,
        next_edge_confidence=0.4,
        jev_tokens=0,
    )


async def run_jev_search(index: RepoIndex, request: InvestigateRequest, judge: JevJudge) -> SearchState:
    budget = request.budget
    frontier, seed_ids, scores = seed_frontier(index, request)
    state = SearchState(seed_scores=scores)
    sem = asyncio.Semaphore(max(1, judge._concurrency))
    async with judge.client() as client:
        while frontier and len(state.judgments) < budget.max_nodes:
            if judge.tokens_used >= budget.max_jev_tokens:
                state.stop_reason = "budget_tokens"
                break
            batch = take_batch(frontier, state, index, request, judge)
            if not batch:
                break

            async def evaluate_item(item: FrontierItem) -> tuple[FrontierItem, Judgment]:
                artifact = index.artifacts[item.artifact_id]
                async with sem:
                    judgment = await judge.evaluate_artifact(
                        question=search_question(request),
                        artifact=artifact,
                        neighbors=neighbor_summary(index, artifact),
                        already_collected=collected_index(index, state.judgments),
                        client=client,
                    )
                return item, judgment

            results = await asyncio.gather(*[evaluate_item(item) for item in batch])
            for item, judgment in results:
                artifact = index.artifacts[item.artifact_id]
                state.judgments[artifact.id] = judgment
                seed_ids.discard(artifact.id)
                if should_expand(judgment):
                    for neighbor_id, depth, via, score in expand_neighbors(
                        index, artifact, judgment, item.depth, budget
                    ):
                        if neighbor_id in state.visited:
                            continue
                        heapq.heappush(
                            frontier,
                            FrontierItem(priority=-score, artifact_id=neighbor_id, depth=depth, via=via),
                        )

            seeds_done = not seed_ids
            if seeds_done and should_stop(state, request):
                state.gap = await judge.evaluate_gap(
                    search_question(request),
                    collected_index(index, state.judgments),
                    client,
                )
                state.stop_reason = "sufficient_or_confident_sot"
                break

            if seeds_done and len(state.judgments) >= 4 and len(state.judgments) % 4 < max(1, judge._concurrency):
                state.gap = await judge.evaluate_gap(
                    search_question(request),
                    collected_index(index, state.judgments),
                    client,
                )
                if state.gap.evidence_sufficient >= 0.72 and has_source_of_truth(state):
                    if tests_ok(state, request):
                        state.stop_reason = "gap_sufficient"
                        break
                if state.gap.next_expand != "stop":
                    boost_frontier(frontier, index, state, state.gap.next_expand, budget)

            if seeds_done and low_information(state):
                state.stop_reason = "low_information_gain"
                break

        if not state.stop_reason:
            state.stop_reason = "budget_nodes" if len(state.judgments) >= budget.max_nodes else "frontier_empty"
        if state.gap is None and state.judgments:
            state.gap = await judge.evaluate_gap(
                search_question(request),
                collected_index(index, state.judgments),
                client,
            )
    return state


def take_batch(
    frontier: list[FrontierItem],
    state: SearchState,
    index: RepoIndex,
    request: InvestigateRequest,
    judge: JevJudge,
) -> list[FrontierItem]:
    budget = request.budget
    batch: list[FrontierItem] = []
    remaining = budget.max_nodes - len(state.judgments)
    while frontier and len(batch) < min(judge._concurrency, remaining):
        item = heapq.heappop(frontier)
        if item.artifact_id in state.visited or item.depth > budget.max_depth:
            continue
        artifact = index.artifacts.get(item.artifact_id)
        if artifact is None:
            continue
        if artifact.kind == "generated" and not request.evidence_policy.allow_generated:
            state.visited.add(item.artifact_id)
            continue
        state.visited.add(item.artifact_id)
        state.files_touched.add(artifact.path)
        batch.append(item)
    return batch


def should_expand(judgment: Judgment) -> bool:
    if judgment.role == "noise" and judgment.relevance < 1.2:
        return False
    if judgment.next_edge == "stop" and judgment.next_edge_confidence >= 0.7 and judgment.relevance < 2:
        return False
    return judgment.relevance >= 1.0 or judgment.is_source_of_truth >= 0.45


def expand_neighbors(
    index: RepoIndex,
    artifact: Artifact,
    judgment: Judgment,
    depth: int,
    budget: Budget,
) -> list[tuple[str, int, str, float]]:
    wanted = EDGE_TYPE_BY_NEXT.get(judgment.next_edge, set())
    if not wanted:
        wanted = {"calls", "tested_by", "documented_by", "contains"}
    results: list[tuple[str, int, str, float]] = []
    outgoing = index.outgoing.get(artifact.id, [])
    incoming = index.incoming.get(artifact.id, [])
    candidates: list[tuple[str, str, str]] = []
    for edge in outgoing:
        if artifact.kind == "test" and edge.type == "calls":
            candidates.append((edge.target, edge.type, "out"))
            continue
        if edge.type in wanted or judgment.next_edge in {"callees", "tests", "docs", "config"}:
            if judgment.next_edge == "callers" and edge.type == "calls":
                continue
            candidates.append((edge.target, edge.type, "out"))
    for edge in incoming:
        if judgment.next_edge == "callers" and edge.type == "calls":
            candidates.append((edge.source, edge.type, "in"))
        elif edge.type in wanted and judgment.next_edge != "callees":
            candidates.append((edge.source, edge.type, "in"))

    ranked: list[tuple[str, str]] = []
    seen: set[str] = set()
    for target, edge_type, _direction in candidates:
        if target in seen or target == artifact.id:
            continue
        seen.add(target)
        ranked.append((target, edge_type))

    for target, edge_type in ranked[: budget.max_branch]:
        neighbor = index.artifacts.get(target)
        if neighbor is None:
            continue
        score = 1.0 + judgment.relevance
        if neighbor.kind == "test" and judgment.next_edge == "tests":
            score += 2
        if neighbor.kind == "docs" and judgment.next_edge == "docs":
            score += 2
        if neighbor.kind == "config" and judgment.next_edge == "config":
            score += 2
        if neighbor.kind in {"function", "module"} and artifact.kind == "test":
            score += 1.5
        results.append((target, depth + 1, edge_type, score))
    return results


def boost_frontier(
    frontier: list[FrontierItem],
    index: RepoIndex,
    state: SearchState,
    next_expand: NextEdge,
    budget: Budget,
) -> None:
    wanted = EDGE_TYPE_BY_NEXT.get(next_expand, set())
    if not wanted:
        return
    added = 0
    for artifact_id, judgment in state.judgments.items():
        if judgment.relevance < 1.5:
            continue
        artifact = index.artifacts[artifact_id]
        fake = Judgment(**{**judgment.__dict__, "next_edge": next_expand})
        for neighbor_id, depth, via, score in expand_neighbors(index, artifact, fake, 1, budget):
            if neighbor_id in state.visited:
                continue
            heapq.heappush(frontier, FrontierItem(priority=-(score + 1.5), artifact_id=neighbor_id, depth=depth, via=via))
            added += 1
            if added >= budget.max_branch:
                return


def should_stop(state: SearchState, request: InvestigateRequest) -> bool:
    if not has_source_of_truth(state):
        return False
    has_test = any(item.role == "test" and item.relevance >= 1.5 for item in state.judgments.values())
    if has_test:
        return True
    if request.evidence_policy.require_tests:
        return False
    return False


def has_source_of_truth(state: SearchState) -> bool:
    for judgment in state.judgments.values():
        if judgment.role == "source_of_truth" and (
            judgment.role_confidence >= 0.55 or judgment.is_source_of_truth >= 0.7
        ):
            if judgment.relevance >= 2.0:
                return True
        if judgment.is_source_of_truth >= 0.82 and judgment.relevance >= 2.0:
            return True
    return False


def tests_ok(state: SearchState, request: InvestigateRequest) -> bool:
    has_test = any(item.role == "test" and item.relevance >= 1.5 for item in state.judgments.values())
    if request.evidence_policy.require_tests:
        return has_test
    return True


def low_information(state: SearchState) -> bool:
    if len(state.judgments) < 6:
        return False
    recent = list(state.judgments.values())[-3:]
    return all(item.role == "noise" and item.relevance < 1.2 for item in recent)


def neighbor_summary(index: RepoIndex, artifact: Artifact) -> dict[str, list[str]]:
    callees = [edge.target for edge in index.outgoing.get(artifact.id, []) if edge.type == "calls"]
    callers = [edge.source for edge in index.incoming.get(artifact.id, []) if edge.type == "calls"]
    tests = [edge.target for edge in index.outgoing.get(artifact.id, []) if edge.type == "tested_by"]
    docs = [edge.target for edge in index.outgoing.get(artifact.id, []) if edge.type == "documented_by"]
    return {
        "callees": [short_id(index, item) for item in callees[:8]],
        "callers": [short_id(index, item) for item in callers[:8]],
        "tests": [short_id(index, item) for item in tests[:8]],
        "docs": [short_id(index, item) for item in docs[:8]],
    }


def collected_index(index: RepoIndex, judgments: dict[str, Judgment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact_id, judgment in judgments.items():
        artifact = index.artifacts[artifact_id]
        rows.append(
            {
                "path": artifact.path,
                "symbol": artifact.symbol,
                "role": judgment.role,
                "relevance": round(judgment.relevance, 2),
                "is_source_of_truth": round(judgment.is_source_of_truth, 2),
                "excerpt": artifact.excerpt[:400],
            }
        )
    return rows


def short_id(index: RepoIndex, artifact_id: str) -> str:
    artifact = index.artifacts.get(artifact_id)
    if artifact is None:
        return artifact_id
    if artifact.symbol:
        return f"{artifact.path}::{artifact.symbol}"
    return artifact.path
