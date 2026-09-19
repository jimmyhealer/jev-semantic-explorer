from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Role = Literal[
    "source_of_truth",
    "implementation",
    "test",
    "caller",
    "config",
    "docs",
    "generated",
    "noise",
]

PacketStatus = Literal[
    "sufficient",
    "insufficient_evidence",
    "unknown",
    "conflict",
]

NextEdge = Literal["callees", "callers", "tests", "docs", "config", "stop"]


@dataclass
class Budget:
    max_nodes: int = 24
    max_depth: int = 3
    max_branch: int = 4
    max_jev_tokens: int = 80_000
    max_cost: int | None = None


@dataclass
class Scope:
    paths: list[str] = field(default_factory=list)
    language: str = "any"
    symbols: list[str] = field(default_factory=list)


@dataclass
class EvidencePolicy:
    require_tests: bool = False
    require_docs: bool = False
    allow_generated: bool = False
    redaction: Literal["strict", "default"] = "default"


@dataclass
class InvestigateRequest:
    question: str
    repo: str
    scope: Scope = field(default_factory=Scope)
    budget: Budget = field(default_factory=Budget)
    evidence_policy: EvidencePolicy = field(default_factory=EvidencePolicy)
    use_jev: bool = True
    index_path: str | None = None


@dataclass
class Artifact:
    id: str
    path: str
    symbol: str | None
    kind: str
    range: tuple[int, int]
    signature: str
    excerpt: str
    docstring: str | None = None
    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    qualname: str | None = None

    def to_state(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "range": [self.range[0], self.range[1]],
            "signature": self.signature,
            "excerpt": self.excerpt,
            "docstring": self.docstring,
            "calls": self.calls[:12],
            "imports": self.imports[:12],
        }


@dataclass
class Edge:
    source: str
    target: str
    type: str


@dataclass
class RepoIndex:
    root: str
    language: str
    artifacts: dict[str, Artifact]
    edges: list[Edge]
    outgoing: dict[str, list[Edge]] = field(default_factory=dict)
    incoming: dict[str, list[Edge]] = field(default_factory=dict)
    by_symbol: dict[str, list[str]] = field(default_factory=dict)
    by_path: dict[str, list[str]] = field(default_factory=dict)
    retrieval: Any = None

    def artifact(self, artifact_id: str) -> Artifact:
        return self.artifacts[artifact_id]


@dataclass
class Judgment:
    artifact_id: str
    role: Role
    role_confidence: float
    role_probabilities: dict[str, float]
    relevance: float
    relevance_confidence: float
    is_source_of_truth: float
    next_edge: NextEdge
    next_edge_confidence: float
    jev_tokens: int = 0


@dataclass
class Citation:
    path: str
    symbol: str | None
    range: list[int]
    why: str
    role: Role
    relevance: float
    confidence: float
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "symbol": self.symbol,
            "range": self.range,
            "why": self.why,
            "role": self.role,
            "relevance": self.relevance,
            "confidence": self.confidence,
        }
        if self.artifact_id:
            payload["artifact_id"] = self.artifact_id
        return payload


@dataclass
class SupportingEvidence:
    tests: list[str] = field(default_factory=list)
    callers: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    config: list[str] = field(default_factory=list)
    implementation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePacket:
    question: str
    scope: dict[str, Any]
    budget: dict[str, Any]
    source_of_truth: list[Citation]
    supporting: SupportingEvidence
    impact_paths: list[str]
    unknowns: list[str]
    alternative_paths: list[str]
    status: PacketStatus
    policy: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    regions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "scope": self.scope,
            "budget": self.budget,
            "source_of_truth": [item.to_dict() for item in self.source_of_truth],
            "supporting": self.supporting.to_dict(),
            "impact_paths": self.impact_paths,
            "unknowns": self.unknowns,
            "alternative_paths": self.alternative_paths,
            "status": self.status,
            "policy": self.policy,
            "metrics": self.metrics,
            "regions": self.regions,
        }
