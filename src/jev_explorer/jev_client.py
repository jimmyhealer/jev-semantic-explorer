from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from jev_explorer.config import api_key
from jev_explorer.models import Artifact, Judgment, NextEdge, Role
from jev_explorer.redaction import redact

ROLE_CRITERIA = {
    "source_of_truth": "This artifact is the actual implementation that defines the behavior asked about.",
    "implementation": "Related implementation that is not the authoritative definition.",
    "test": "A test that verifies the behavior asked about.",
    "caller": "Code that calls or is affected by the behavior asked about.",
    "config": "Settings, feature flags, constants, or schema that configure the behavior.",
    "docs": "Documentation that describes the behavior.",
    "generated": "Generated or derived code, not a source of truth.",
    "noise": "Unrelated to the question.",
}

RELEVANCE_CRITERIA = [
    "Unrelated and can be ignored.",
    "Background that helps understanding but is not evidence.",
    "Related; changing it could affect the answer.",
    "Core; the question cannot be answered without it.",
]

NEXT_EDGE_CRITERIA = {
    "callees": "Need functions this artifact calls.",
    "callers": "Need places that call this artifact.",
    "tests": "Need tests that cover this behavior.",
    "docs": "Need documentation about this behavior.",
    "config": "Need config, flags, or schema.",
    "stop": "No further expansion is useful.",
}


@dataclass
class GapJudgment:
    evidence_sufficient: float
    tests_consistent: float
    docs_consistent: float
    next_expand: NextEdge
    next_expand_confidence: float
    jev_tokens: int = 0


class JevJudge:
    def __init__(self, concurrency: int = 6) -> None:
        self._key = api_key()
        self._concurrency = concurrency
        self._lock = asyncio.Lock()
        self.tokens_used = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    async def evaluate_artifact(
        self,
        question: str,
        artifact: Artifact,
        neighbors: dict[str, list[str]],
        already_collected: list[dict[str, Any]],
        client: AsyncTypeSafeClient,
    ) -> Judgment:
        state = {
            "question": question,
            "artifact": artifact.to_state(),
            "neighbors": neighbors,
            "already_collected": already_collected[:12],
        }
        questions = {
            "role": Choice(
                instructions=(
                    "Given `question`, what role does `artifact` play? "
                    "Use `artifact.excerpt` and `artifact.signature`. "
                    "Choose noise if it does not help answer `question`."
                ),
                criteria=ROLE_CRITERIA,
            ),
            "relevance": Score(
                instructions=(
                    "How necessary is `artifact` for answering `question`? "
                    "0 means ignore, 3 means the question cannot be answered without it."
                ),
                criteria=RELEVANCE_CRITERIA,
            ),
            "is_source_of_truth": Noul(
                instructions=(
                    "`artifact.excerpt` is the source of truth that actually defines "
                    "or enforces the behavior asked in `question`. "
                    "True only if this is the authoritative implementation, not a caller, test, doc, or config."
                ),
                criteria={
                    "true": "This excerpt is the actual defining implementation.",
                    "false": "This is not the authoritative implementation.",
                },
            ),
            "next_edge": Choice(
                instructions=(
                    "If more evidence is needed after reading `artifact`, which graph edge should be expanded next? "
                    "Choose stop if this artifact is noise or already sufficient locally."
                ),
                criteria=NEXT_EDGE_CRITERIA,
            ),
        }
        response = await client.system_one(state=redact_state(state), questions=questions)
        tokens = await self._record_usage(response)
        role = response.choices["role"]
        relevance = response.scores["relevance"]
        next_edge = response.choices["next_edge"]
        return Judgment(
            artifact_id=artifact.id,
            role=normalize_role(role.choice),
            role_confidence=float(role.confidence or 0.0),
            role_probabilities={key: float(value) for key, value in (role.probabilities or {}).items()},
            relevance=float(relevance.score),
            relevance_confidence=float(relevance.confidence or 0.0),
            is_source_of_truth=float(response.nouls["is_source_of_truth"].noul),
            next_edge=normalize_next_edge(next_edge.choice),
            next_edge_confidence=float(next_edge.confidence or 0.0),
            jev_tokens=tokens,
        )

    async def evaluate_gap(
        self,
        question: str,
        collected: list[dict[str, Any]],
        client: AsyncTypeSafeClient,
    ) -> GapJudgment:
        state = {"question": question, "collected_evidence": collected}
        questions = {
            "evidence_sufficient": Noul(
                instructions=(
                    "The items in `collected_evidence` are already enough to locate the source of truth "
                    "for `question` and to start reasoning about it."
                ),
                criteria={
                    "true": "A knowledgeable engineer could stop exploring and act on this set.",
                    "false": "Important implementation, tests, or callers are still missing.",
                },
            ),
            "tests_consistent": Noul(
                instructions=(
                    "The tests in `collected_evidence`, if any, are consistent with the listed implementation "
                    "for `question`. If no tests are present, answer false."
                ),
            ),
            "docs_consistent": Noul(
                instructions=(
                    "The docs in `collected_evidence`, if any, describe the same behavior as the implementation. "
                    "If no docs are present, answer false."
                ),
            ),
            "next_expand": Choice(
                instructions=(
                    "If evidence is still insufficient, which edge type should be expanded next across the graph?"
                ),
                criteria=NEXT_EDGE_CRITERIA,
            ),
        }
        response = await client.system_one(state=redact_state(state), questions=questions)
        tokens = await self._record_usage(response)
        nxt = response.choices["next_expand"]
        return GapJudgment(
            evidence_sufficient=float(response.nouls["evidence_sufficient"].noul),
            tests_consistent=float(response.nouls["tests_consistent"].noul),
            docs_consistent=float(response.nouls["docs_consistent"].noul),
            next_expand=normalize_next_edge(nxt.choice),
            next_expand_confidence=float(nxt.confidence or 0.0),
            jev_tokens=tokens,
        )

    def client(self) -> AsyncTypeSafeClient:
        return AsyncTypeSafeClient(api_key=self._key, timeout=60.0)

    async def _record_usage(self, response: Any) -> int:
        inp, out = usage_breakdown(response)
        total = inp + out
        async with self._lock:
            self.input_tokens += inp
            self.output_tokens += out
            self.tokens_used += total
            self.calls += 1
        return total


def redact_state(state: dict[str, Any]) -> dict[str, Any]:
    return _redact_value(state)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    return value


def usage_breakdown(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def usage_tokens(response: Any) -> int:
    inp, out = usage_breakdown(response)
    return inp + out


def normalize_role(value: str | None) -> Role:
    allowed = {
        "source_of_truth",
        "implementation",
        "test",
        "caller",
        "config",
        "docs",
        "generated",
        "noise",
    }
    if value in allowed:
        return value  # type: ignore[return-value]
    return "noise"


def normalize_next_edge(value: str | None) -> NextEdge:
    allowed = {"callees", "callers", "tests", "docs", "config", "stop"}
    if value in allowed:
        return value  # type: ignore[return-value]
    return "stop"
