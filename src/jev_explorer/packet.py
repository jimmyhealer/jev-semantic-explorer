from __future__ import annotations

from jev_explorer.jev_client import GapJudgment
from jev_explorer.models import (
    Citation,
    EvidencePacket,
    InvestigateRequest,
    Judgment,
    PacketStatus,
    RepoIndex,
    SupportingEvidence,
)


def build_packet(
    request: InvestigateRequest,
    index: RepoIndex,
    judgments: dict[str, Judgment],
    gap: GapJudgment | None,
) -> EvidencePacket:
    citations: list[Citation] = []
    for artifact_id, judgment in judgments.items():
        artifact = index.artifacts[artifact_id]
        citations.append(
            Citation(
                path=artifact.path,
                symbol=artifact.symbol,
                range=[artifact.range[0], artifact.range[1]],
                why=make_why(artifact.signature, artifact.excerpt, judgment),
                role=judgment.role,
                relevance=round(judgment.relevance, 3),
                confidence=round(max(judgment.role_confidence, judgment.is_source_of_truth), 3),
                artifact_id=artifact.id,
            )
        )

    source = [
        item
        for item in citations
        if is_source(item, judgments.get(item.artifact_id or ""))
    ]
    source.sort(key=lambda item: (item.relevance, item.confidence), reverse=True)
    source = prefer_symbol_citations(source)

    supporting = SupportingEvidence()
    for item in citations:
        if item in source:
            continue
        if item.relevance < 1.3 or item.role == "noise":
            continue
        path = item.path
        if item.role == "test":
            append_unique(supporting.tests, path)
        elif item.role == "caller":
            append_unique(supporting.callers, path)
        elif item.role == "docs":
            append_unique(supporting.docs, path)
        elif item.role == "config":
            append_unique(supporting.config, path)
        elif item.role in {"implementation", "source_of_truth"}:
            append_unique(supporting.implementation, path)

    attach_graph_support(index, source, supporting)

    impact_paths = build_impact_paths(index, source, judgments)
    unknowns, alternatives, status = decide_status(request, source, supporting, citations, gap)

    return EvidencePacket(
        question=request.question,
        scope={
            "repo": index.root,
            "language": request.scope.language,
            "paths": request.scope.paths,
        },
        budget={
            "max_nodes": request.budget.max_nodes,
            "max_depth": request.budget.max_depth,
            "max_jev_tokens": request.budget.max_jev_tokens,
        },
        source_of_truth=source[:5],
        supporting=supporting,
        impact_paths=impact_paths[:8],
        unknowns=unknowns,
        alternative_paths=alternatives,
        status=status,
        policy={
            "hybrid_retrieval": request.use_jev,
            "redacted": ["secrets", "env", "private_keys"],
            "require_tests": request.evidence_policy.require_tests,
            "require_docs": request.evidence_policy.require_docs,
            "allow_generated": request.evidence_policy.allow_generated,
            "redaction": request.evidence_policy.redaction,
        },
    )


def is_source(citation: Citation, judgment: Judgment | None) -> bool:
    if judgment is None:
        return False
    if citation.role == "source_of_truth" and citation.relevance >= 1.8 and citation.confidence >= 0.55:
        return True
    return judgment.is_source_of_truth >= 0.78 and citation.relevance >= 2.0


def prefer_symbol_citations(source: list[Citation]) -> list[Citation]:
    by_path: dict[str, list[Citation]] = {}
    for item in source:
        by_path.setdefault(item.path, []).append(item)
    preferred: list[Citation] = []
    for items in by_path.values():
        symbols = [item for item in items if item.symbol]
        preferred.extend(symbols or items)
    preferred.sort(key=lambda item: (item.relevance, item.confidence), reverse=True)
    return preferred


def make_why(signature: str, excerpt: str, judgment: Judgment) -> str:
    quote = first_code_line(excerpt)
    facts = [f"signature={signature}", f"role={judgment.role}", f"relevance={judgment.relevance:.2f}"]
    if quote:
        facts.append(f"quote={quote}")
    return "; ".join(facts)


def first_code_line(excerpt: str) -> str:
    for line in excerpt.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
            return stripped[:180]
    return ""


def append_unique(bucket: list[str], value: str) -> None:
    if value not in bucket:
        bucket.append(value)


def attach_graph_support(index: RepoIndex, source: list[Citation], supporting: SupportingEvidence) -> None:
    for citation in source:
        artifact_id = citation.artifact_id
        if not artifact_id:
            continue
        for edge in index.outgoing.get(artifact_id, []):
            target = index.artifacts.get(edge.target)
            if target is None:
                continue
            if edge.type == "tested_by":
                append_unique(supporting.tests, target.path)
            elif edge.type == "documented_by":
                append_unique(supporting.docs, target.path)
        for edge in index.incoming.get(artifact_id, []):
            if edge.type == "calls":
                caller = index.artifacts.get(edge.source)
                if caller:
                    append_unique(supporting.callers, caller.path)


def build_impact_paths(index: RepoIndex, source: list[Citation], judgments: dict[str, Judgment]) -> list[str]:
    paths: list[str] = []
    for citation in source:
        if not citation.artifact_id:
            continue
        root_name = citation.symbol or citation.path
        callers = [
            edge.source
            for edge in index.incoming.get(citation.artifact_id, [])
            if edge.type == "calls"
        ]
        if not callers:
            paths.append(root_name)
            continue
        for caller_id in callers[:4]:
            caller = index.artifacts.get(caller_id)
            if caller is None:
                continue
            chain = [root_name, caller.symbol or caller.path]
            second = [
                edge.source
                for edge in index.incoming.get(caller_id, [])
                if edge.type == "calls"
            ]
            if second:
                nxt = index.artifacts.get(second[0])
                if nxt:
                    chain.append(nxt.symbol or nxt.path)
            paths.append(" <- ".join(chain))
    return list(dict.fromkeys(paths))


def decide_status(
    request: InvestigateRequest,
    source: list[Citation],
    supporting: SupportingEvidence,
    citations: list[Citation],
    gap: GapJudgment | None,
) -> tuple[list[str], list[str], PacketStatus]:
    unknowns: list[str] = []
    alternatives: list[str] = []

    if not source:
        unknowns.append("No high-confidence source of truth was located.")
    if request.evidence_policy.require_tests and not supporting.tests:
        unknowns.append("Required tests were not attached to the packet.")
    if request.evidence_policy.require_docs and not supporting.docs:
        unknowns.append("Required docs were not attached to the packet.")
    if source and not supporting.tests:
        unknowns.append("No test covering the source of truth was included.")

    judged_roles = {item.role for item in citations}
    if gap is not None:
        if gap.docs_consistent < 0.35 and supporting.docs and "docs" in judged_roles:
            unknowns.append("Docs may drift from implementation.")
        if gap.tests_consistent < 0.35 and supporting.tests and "test" in judged_roles:
            unknowns.append("Tests may not match the implementation.")
        if gap.evidence_sufficient < 0.5:
            unknowns.append("Jev judged the collected evidence incomplete.")
        if gap.next_expand != "stop" and gap.evidence_sufficient < 0.72:
            alternatives.append(f"Expand {gap.next_expand} from current nodes.")

    leftover = [
        item
        for item in citations
        if item not in source and item.relevance >= 2.0 and item.role not in {"noise", "generated"}
    ]
    for item in leftover[:4]:
        label = item.symbol or item.path
        alternatives.append(f"Also inspect {item.path}::{label} (relevance={item.relevance:.2f}, role={item.role}).")

    if gap is not None and gap.docs_consistent < 0.3 and supporting.docs and source:
        status: PacketStatus = "conflict"
    elif not source and not any(item.relevance >= 1.5 for item in citations):
        status = "unknown"
    elif not source or (gap is not None and gap.evidence_sufficient < 0.55):
        status = "insufficient_evidence"
    else:
        status = "sufficient"
        if request.evidence_policy.require_tests and not supporting.tests:
            status = "insufficient_evidence"
    return unknowns, alternatives, status
