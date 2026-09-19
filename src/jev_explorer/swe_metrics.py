from __future__ import annotations

from typing import Any

Region = tuple[str, int, int]


def _as_region(item: Any) -> Region:
    if isinstance(item, dict):
        path = item.get("path") or item.get("file")
        start = int(item.get("start") or item.get("start_line") or 1)
        end = int(item.get("end") or item.get("end_line") or start)
        return str(path), start, end
    return str(item[0]), int(item[1]), int(item[2])


def _lines(regions: list[Any]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for item in regions:
        path, start, end = _as_region(item)
        if not path:
            continue
        lo, hi = sorted((max(1, start), max(1, end)))
        for line in range(lo, hi + 1):
            out.add((path, line))
    return out


def _files(regions: list[Any]) -> set[str]:
    return { _as_region(item)[0] for item in regions if _as_region(item)[0] }


def _overlap(a: Any, b: Any) -> bool:
    pa, sa, ea = _as_region(a)
    pb, sb, eb = _as_region(b)
    if pa != pb:
        return False
    lo_a, hi_a = sorted((sa, ea))
    lo_b, hi_b = sorted((sb, eb))
    return lo_a <= hi_b and lo_b <= hi_a


def optional_regions(gt: dict[str, Any]) -> list[Any]:
    mapped = gt.get("read_optional_regions_map") or {}
    out: list[Any] = []
    for regions in mapped.values():
        out.extend(regions or [])
    return out


def optional_files(gt: dict[str, Any]) -> set[str]:
    mapped = gt.get("read_optional_files_map") or {}
    out: set[str] = set()
    for files in mapped.values():
        out.update(files or [])
    return out


def score_prediction(preds: list[Any], gt: dict[str, Any]) -> dict[str, float]:
    core_regions = gt.get("read_core_regions") or []
    core_files = set(gt.get("read_core_files") or _files(core_regions))
    pred_lines = _lines(preds)
    core_lines = _lines(core_regions)
    opt_lines = _lines(optional_regions(gt))
    pred_files = _files(preds)

    precision = (len(pred_lines & core_lines) / len(pred_lines)) if pred_lines else 0.0
    recall = (len(pred_lines & core_lines) / len(core_lines)) if core_lines else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    hit_file = (len(pred_files & core_files) / len(core_files)) if core_files else 0.0
    useful = pred_files - core_files - optional_files(gt)
    noise_file = (len(useful) / len(pred_files)) if pred_files else 0.0
    if core_regions:
        hit_region = sum(
            1 for core in core_regions if any(_overlap(core, pred) for pred in preds)
        ) / len(core_regions)
    else:
        hit_region = 0.0
    useful_lines = pred_lines & (core_lines | opt_lines)
    context_efficiency = (len(useful_lines) / len(pred_lines)) if pred_lines else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "hit_file_rate": round(hit_file, 4),
        "noise_file_rate": round(noise_file, 4),
        "hit_region_rate": round(hit_region, 4),
        "context_efficiency": round(context_efficiency, 4),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    numeric_keys = [
        key
        for key in rows[0]
        if all(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool) for row in rows)
    ]
    return {key: round(sum(float(row[key]) for row in rows) / len(rows), 4) for key in numeric_keys}
