from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jev_explorer.models import Judgment, RepoIndex


@dataclass
class RankedRegion:
    path: str
    start: int
    end: int
    score: float
    role: str
    symbol: str | None = None
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start": int(self.start),
            "end": int(self.end),
            "score": round(self.score, 4),
            "role": self.role,
            "symbol": self.symbol,
            "artifact_id": self.artifact_id,
        }

    def line_count(self) -> int:
        return max(int(self.end) - int(self.start) + 1, 1)


def ranked_regions(
    index: RepoIndex,
    judgments: dict[str, Judgment],
    k: int = 5,
    line_budget: int = 500,
    seed_scores: dict[str, float] | None = None,
) -> list[RankedRegion]:
    """Turn every judged artifact into SWE-Explore-style ranked regions."""
    del seed_scores
    scored: list[RankedRegion] = []
    for artifact_id, judgment in judgments.items():
        artifact = index.artifacts.get(artifact_id)
        if artifact is None:
            continue
        if judgment.role == "noise" and judgment.relevance < 1.3:
            continue
        sot_boost = 2.0 if judgment.role == "source_of_truth" or judgment.is_source_of_truth >= 0.7 else 1.0
        score = judgment.relevance * max(judgment.role_confidence, judgment.is_source_of_truth, 0.15) * sot_boost
        if score <= 0:
            continue
        start, end = artifact.range
        scored.append(
            RankedRegion(
                path=artifact.path,
                start=int(start),
                end=int(end),
                score=float(score),
                role=judgment.role,
                symbol=artifact.symbol,
                artifact_id=artifact.id,
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    merged = merge_overlaps(scored)
    selected: list[RankedRegion] = []
    used_lines = 0
    for region in merged:
        if len(selected) >= k:
            break
        span = region.line_count()
        if selected and used_lines + span > line_budget:
            continue
        selected.append(region)
        used_lines += span
    return selected


def merge_overlaps(regions: list[RankedRegion]) -> list[RankedRegion]:
    by_path: dict[str, list[RankedRegion]] = {}
    for region in regions:
        by_path.setdefault(region.path, []).append(region)
    merged: list[RankedRegion] = []
    for path, group in by_path.items():
        group.sort(key=lambda item: (item.start, item.end))
        current: RankedRegion | None = None
        for region in group:
            if current is None:
                current = region
                continue
            if region.start <= current.end + 1:
                current = RankedRegion(
                    path=path,
                    start=current.start,
                    end=max(current.end, region.end),
                    score=max(current.score, region.score),
                    role=current.role if current.score >= region.score else region.role,
                    symbol=current.symbol or region.symbol,
                    artifact_id=current.artifact_id or region.artifact_id,
                )
            else:
                merged.append(current)
                current = region
        if current is not None:
            merged.append(current)
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged


def swe_explore_prediction(regions: list[RankedRegion]) -> list[dict[str, Any]]:
    """Official-style list: path plus inclusive line range, ranked."""
    return [
        {
            "file": region.path,
            "start_line": region.start,
            "end_line": region.end,
            "score": region.score,
        }
        for region in regions
    ]
