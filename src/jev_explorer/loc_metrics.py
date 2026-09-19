from __future__ import annotations

from typing import Any


def patch_files(row: dict[str, Any]) -> list[str]:
    files = row.get("edit_files") or row.get("gold_files")
    if files:
        return list(files)
    patch = row.get("patch") or ""
    found: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            found.append(line[6:])
        elif line.startswith("diff --git"):
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                found.append(parts[1])
    return list(dict.fromkeys(found))


def edit_functions(row: dict[str, Any]) -> list[str]:
    funcs = row.get("edit_functions") or row.get("gold_functions") or []
    if isinstance(funcs, str):
        return [funcs]
    out: list[str] = []
    for item in funcs:
        if isinstance(item, dict):
            name = item.get("function") or item.get("name") or item.get("qualname")
            if name:
                out.append(str(name))
        else:
            out.append(str(item))
    return out


def acc_at_k(predicted: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 1.0
    top = set(predicted[:k])
    return len(top & set(gold)) / len(set(gold))


def score_loc(predicted_files: list[str], predicted_funcs: list[str], row: dict[str, Any]) -> dict[str, float]:
    gold_files = patch_files(row)
    gold_funcs = edit_functions(row)
    return {
        "acc_file@1": round(acc_at_k(predicted_files, gold_files, 1), 4),
        "acc_file@5": round(acc_at_k(predicted_files, gold_files, 5), 4),
        "acc_func@5": round(acc_at_k(predicted_funcs, gold_funcs, 5), 4) if gold_funcs else None,  # type: ignore[dict-item]
        "gold_files": len(gold_files),
        "gold_funcs": len(gold_funcs),
    }
