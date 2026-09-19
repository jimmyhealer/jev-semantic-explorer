from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from jev_explorer.gitignore import GitIgnore
from jev_explorer.models import Artifact, Edge, RepoIndex
from jev_explorer.redaction import (
    looks_like_injection,
    redact,
    should_skip_dir,
    should_skip_file,
)
from jev_explorer.retrieval import build_retrieval

COMMON_CALLEES = {
    "print",
    "len",
    "range",
    "str",
    "int",
    "float",
    "dict",
    "list",
    "set",
    "tuple",
    "bool",
    "isinstance",
    "issubclass",
    "getattr",
    "setattr",
    "hasattr",
    "super",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "min",
    "max",
    "sum",
    "open",
    "format",
    "join",
    "split",
    "append",
    "extend",
    "items",
    "keys",
    "values",
    "get",
    "now",
    "time",
    "replace",
    "strip",
    "lower",
    "upper",
    "path",
    "exists",
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "require",
    "include",
}

DOC_SUFFIXES = {".md", ".rst", ".txt", ".adoc"}
CONFIG_HINTS = ("config", "settings", "flags", "feature_flag")
GENERATED_HINTS = ("generated", "gen/", "_pb2.py", ".g.py", ".pb.go", "_generated.")
MAX_CHUNK_LINES = 48
MAX_FILE_BYTES = 1_000_000

# What a coding agent actually greps. Not GitHub issues/wiki APIs.
SCAN_SUFFIXES = {
    *DOC_SUFFIXES,
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".kts",
    ".swift",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".cs",
    ".scala",
    ".clj",
    ".ex",
    ".exs",
    ".erl",
    ".hs",
    ".lua",
    ".r",
    ".jl",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".vue",
    ".svelte",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".graphql",
    ".gql",
    ".proto",
    ".zig",
    ".nim",
    ".ml",
    ".mli",
    ".dart",
    ".m",
    ".mm",
}

DECL_RE = re.compile(
    r"""
    ^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:pub(?:lic)?\s+)?(?:static\s+)?
    (?:
        def | class | fn | func | function | impl | interface |
        struct | enum | trait | module | namespace | record
    )
    \s+([A-Za-z_][\w]*)
    |
    ^[ \t]*(?:export\s+)?(?:const|let|var|val)\s+([A-Za-z_][\w]*)\s*=\s*(?:async\s+)?(?:function|\()
    |
    ^[ \t]*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)
    |
    ^[ \t]*(?:public|private|protected|internal|static|[\s])*
    (?:class|interface|enum|record|struct)\s+([A-Za-z_][\w]*)
    |
    ^(?P<hashes>\#{1,6})\s+(?P<heading>.+?)\s*$
    """,
    re.VERBOSE,
)
CALL_RE = re.compile(r"\b([A-Za-z_][\w]*)\s*\(")
IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+)|use\s+([\w:]+)|"
    r"#include\s+[<\"]([\w./]+)[>\"]|require\(\s*['\"]([^'\"]+))"
)

CLASS_WORDS = {"class", "struct", "enum", "interface", "trait", "record", "type", "impl"}
FN_WORDS = {"def", "fn", "func", "function"}


def build_index(root: str | Path, scope_paths: list[str] | None = None) -> RepoIndex:
    root_path = Path(root).resolve()
    artifacts: dict[str, Artifact] = {}
    edges: list[Edge] = []

    for file_path in iter_source_files(root_path, scope_paths):
        rel = file_path.relative_to(root_path).as_posix()
        text = read_text(file_path)
        file_arts, file_edges = index_file(rel, text)
        artifacts.update(file_arts)
        edges.extend(file_edges)

    add_call_edges(artifacts, edges)
    add_test_edges(artifacts, edges)
    add_doc_edges(artifacts, edges)
    index = RepoIndex(root=str(root_path), language="any", artifacts=artifacts, edges=edges)
    return finalize_index(index)


def finalize_index(index: RepoIndex) -> RepoIndex:
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    incoming: dict[str, list[Edge]] = defaultdict(list)
    for edge in index.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    by_symbol: dict[str, list[str]] = defaultdict(list)
    by_path: dict[str, list[str]] = defaultdict(list)
    for artifact in index.artifacts.values():
        by_path[artifact.path].append(artifact.id)
        if artifact.symbol:
            by_symbol[artifact.symbol].append(artifact.id)
            if artifact.qualname and artifact.qualname != artifact.symbol:
                by_symbol[artifact.qualname].append(artifact.id)

    index.outgoing = dict(outgoing)
    index.incoming = dict(incoming)
    index.by_symbol = {key: list(dict.fromkeys(value)) for key, value in by_symbol.items()}
    index.by_path = dict(by_path)
    build_retrieval(index)
    return index


def dump_index(index: RepoIndex, destination: str | Path) -> None:
    payload = {
        "root": index.root,
        "language": index.language,
        "artifacts": {
            key: {
                "id": art.id,
                "path": art.path,
                "symbol": art.symbol,
                "kind": art.kind,
                "range": list(art.range),
                "signature": art.signature,
                "excerpt": art.excerpt,
                "docstring": art.docstring,
                "calls": art.calls,
                "imports": art.imports,
                "qualname": art.qualname,
            }
            for key, art in index.artifacts.items()
        },
        "edges": [edge.__dict__ for edge in index.edges],
    }
    Path(destination).write_text(json.dumps(payload), encoding="utf-8")


def load_index(source: str | Path) -> RepoIndex:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    artifacts = {
        key: Artifact(
            id=raw["id"],
            path=raw["path"],
            symbol=raw.get("symbol"),
            kind=raw["kind"],
            range=(int(raw["range"][0]), int(raw["range"][1])),
            signature=raw.get("signature", raw["path"]),
            excerpt=raw.get("excerpt", ""),
            docstring=raw.get("docstring"),
            calls=raw.get("calls") or [],
            imports=raw.get("imports") or [],
            qualname=raw.get("qualname"),
        )
        for key, raw in payload["artifacts"].items()
    }
    edges = [
        Edge(source=item["source"], target=item["target"], type=item["type"])
        for item in payload.get("edges", [])
    ]
    index = RepoIndex(
        root=payload.get("root", str(Path(source).resolve().parent)),
        language=payload.get("language", "any"),
        artifacts=artifacts,
        edges=edges,
    )
    return finalize_index(index)


_INDEX_CACHE: dict[str, RepoIndex] = {}
MAX_DUMP_BYTES = 40 * 1024 * 1024


def get_or_build_index(
    root: str | Path,
    scope_paths: list[str] | None = None,
    index_path: str | None = None,
) -> RepoIndex:
    root_path = Path(root).resolve()
    cache_key = str(root_path)
    if cache_key in _INDEX_CACHE:
        return _INDEX_CACHE[cache_key]
    candidate = Path(index_path) if index_path else root_path / ".jev-index.json"
    if candidate.is_file() and candidate.stat().st_size <= MAX_DUMP_BYTES:
        loaded = load_index(candidate)
        _INDEX_CACHE[cache_key] = loaded
        return loaded
    index = build_index(root_path, scope_paths)
    _INDEX_CACHE[cache_key] = index
    return index


def iter_source_files(root: Path, scope_paths: list[str] | None = None) -> list[Path]:
    allowed_prefixes: list[str] = []
    if scope_paths:
        allowed_prefixes = [normalize_scope(item) for item in scope_paths]
    ignored = GitIgnore.from_root(root)

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = [
            name
            for name in dirnames
            if not should_skip_dir(name)
            and not ignored.matches(
                name if rel_dir in {".", ""} else f"{rel_dir}/{name}",
                is_dir=True,
            )
        ]
        for name in filenames:
            if should_skip_file(name):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            if ignored.matches(rel, is_dir=False):
                continue
            if allowed_prefixes and not any(
                rel == prefix or rel.startswith(prefix.rstrip("/") + "/")
                for prefix in allowed_prefixes
            ):
                continue
            files.append(path)
    return sorted(files)


def normalize_scope(item: str) -> str:
    return item.replace("\\", "/").lstrip("./")


def read_text(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if b"\x00" in data[:8192]:
        return ""
    return data.decode("utf-8", errors="replace")


def index_file(rel: str, text: str) -> tuple[dict[str, Artifact], list[Edge]]:
    lines = text.splitlines()
    file_kind = classify_file(rel, text)
    imports = collect_imports(text)
    artifacts: dict[str, Artifact] = {}
    edges: list[Edge] = []

    file_id = artifact_id(rel, None)
    artifacts[file_id] = Artifact(
        id=file_id,
        path=rel,
        symbol=None,
        kind=file_kind,
        range=(1, max(len(lines), 1)),
        signature=rel,
        excerpt=make_excerpt(lines, 1, min(len(lines), 40)),
        docstring=None,
        imports=imports,
    )

    starts = declaration_starts(lines)
    if not starts:
        if file_kind == "docs":
            for start, end, heading in heading_windows(lines):
                art_id = artifact_id(rel, heading)
                artifacts[art_id] = Artifact(
                    id=art_id,
                    path=rel,
                    symbol=heading,
                    kind="docs",
                    range=(start, end),
                    signature=heading,
                    excerpt=make_excerpt(lines, start, end),
                    docstring=heading,
                    imports=imports,
                    qualname=heading,
                )
                edges.append(Edge(source=file_id, target=art_id, type="contains"))
        return artifacts, edges

    for i, (start, symbol, kind) in enumerate(starts):
        end = (starts[i + 1][0] - 1) if i + 1 < len(starts) else len(lines)
        end = min(end, start + MAX_CHUNK_LINES - 1)
        end = max(end, start)
        excerpt = make_excerpt(lines, start, end)
        art_id = artifact_id(rel, symbol)
        if art_id in artifacts:
            art_id = f"{art_id}@{start}"
        chunk_kind = "generated" if file_kind == "generated" else kind
        if file_kind == "test" and chunk_kind == "function":
            chunk_kind = "test"
        artifacts[art_id] = Artifact(
            id=art_id,
            path=rel,
            symbol=symbol,
            kind=chunk_kind,
            range=(start, end),
            signature=lines[start - 1].strip()[:240] or symbol,
            excerpt=excerpt,
            docstring=first_comment(lines, start, end),
            calls=collect_calls(excerpt, symbol),
            imports=imports,
            qualname=symbol,
        )
        edges.append(Edge(source=file_id, target=art_id, type="contains"))
    return artifacts, edges


def declaration_starts(lines: list[str]) -> list[tuple[int, str, str]]:
    found: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(lines, start=1):
        match = DECL_RE.match(line)
        if not match:
            continue
        symbol = next((group for group in match.groups() if group), None)
        if match.groupdict().get("heading"):
            symbol = slug(match.group("heading"))
            found.append((lineno, symbol, "docs"))
            continue
        if not symbol:
            continue
        tokens = re.findall(r"[A-Za-z_]+", line[:96])
        if any(tok in FN_WORDS for tok in tokens[:8]):
            kind = "function"
        elif any(tok in CLASS_WORDS for tok in tokens[:8]):
            kind = "class"
        else:
            kind = "function"
        found.append((lineno, symbol, kind))
    return found


def heading_windows(lines: list[str]) -> list[tuple[int, int, str]]:
    starts = [(i, slug(line.lstrip("# ").strip())) for i, line in enumerate(lines, start=1) if line.startswith("#")]
    windows: list[tuple[int, int, str]] = []
    for i, (start, heading) in enumerate(starts):
        end = (starts[i + 1][0] - 1) if i + 1 < len(starts) else min(len(lines), start + MAX_CHUNK_LINES - 1)
        windows.append((start, max(end, start), heading or f"h{start}"))
    return windows


def slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())[:80].strip("_")
    return cleaned or "section"


def classify_file(rel: str, text: str) -> str:
    lowered = rel.lower()
    suffix = Path(rel).suffix.lower()
    if any(hint in lowered for hint in GENERATED_HINTS) or "autogenerated" in text[:400].lower():
        return "generated"
    if is_test_path(rel):
        return "test"
    if suffix in DOC_SUFFIXES:
        return "docs"
    if any(hint in lowered for hint in CONFIG_HINTS) or suffix in {".yml", ".yaml", ".toml", ".json"}:
        return "config"
    return "module"


def is_test_path(rel: str) -> bool:
    name = Path(rel).name.lower()
    return (
        rel.startswith("tests/")
        or "/tests/" in rel
        or "/__tests__/" in rel
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith("_test.go")
        or ".test." in name
        or name.endswith(".spec.ts")
        or name.endswith(".spec.js")
        or name.endswith("_test.rs")
    )


def collect_imports(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines()[:80]:
        match = IMPORT_RE.match(line)
        if not match:
            continue
        raw = next((group for group in match.groups() if group), "")
        names.append(raw.split(".")[0].split("/")[0].split("::")[0])
    return list(dict.fromkeys(names))


def collect_calls(excerpt: str, self_symbol: str | None) -> list[str]:
    names: list[str] = []
    for name in CALL_RE.findall(excerpt):
        if name in COMMON_CALLEES or name == self_symbol:
            continue
        names.append(name)
    return list(dict.fromkeys(names))[:24]


def add_call_edges(artifacts: dict[str, Artifact], edges: list[Edge]) -> None:
    by_symbol: dict[str, list[str]] = defaultdict(list)
    for artifact in artifacts.values():
        if artifact.symbol:
            by_symbol[artifact.symbol].append(artifact.id)
        if artifact.qualname:
            by_symbol[artifact.qualname].append(artifact.id)

    for artifact in artifacts.values():
        for callee in artifact.calls:
            targets = by_symbol.get(callee, [])
            for target in targets:
                if target == artifact.id:
                    continue
                edges.append(Edge(source=artifact.id, target=target, type="calls"))


def add_test_edges(artifacts: dict[str, Artifact], edges: list[Edge]) -> None:
    tests = [art for art in artifacts.values() if art.kind == "test"]
    implementations = [
        art
        for art in artifacts.values()
        if art.kind in {"function", "class", "module"} and art.symbol
    ]
    for test in tests:
        blob = f"{test.path} {test.symbol or ''} {test.excerpt} {test.docstring or ''}"
        for impl in implementations:
            if impl.symbol and impl.symbol in blob:
                edges.append(Edge(source=impl.id, target=test.id, type="tested_by"))


def add_doc_edges(artifacts: dict[str, Artifact], edges: list[Edge]) -> None:
    docs = [art for art in artifacts.values() if art.kind == "docs"]
    implementations = [
        art for art in artifacts.values() if art.kind in {"function", "class", "module", "config"}
    ]
    for doc in docs:
        blob = f"{doc.path} {doc.excerpt}".lower()
        for impl in implementations:
            tokens = [impl.path.lower()]
            if impl.symbol:
                tokens.append(impl.symbol.lower())
            if any(token and token in blob for token in tokens):
                edges.append(Edge(source=impl.id, target=doc.id, type="documented_by"))


def artifact_id(path: str, symbol: str | None) -> str:
    return f"{path}::{symbol}" if symbol else path


def first_comment(lines: list[str], start: int, end: int) -> str | None:
    for line in lines[start - 1 : min(end, start + 8)]:
        stripped = line.strip()
        if stripped.startswith(('"""', "'''")):
            return stripped.strip("\"'")[:240] or None
        if stripped.startswith(("#", "//", "///", "--")):
            return stripped.lstrip("#/!- ").strip()[:240] or None
        if stripped.startswith("*") and not stripped.startswith("*/"):
            return stripped.lstrip("* ").strip()[:240] or None
    return None


def make_excerpt(lines: list[str], start: int, end: int) -> str:
    chunk = lines[max(start - 1, 0) : end]
    text = "\n".join(chunk)
    text = redact(text)
    if looks_like_injection(text):
        text = "[comment redacted: possible instruction injection]\n" + "\n".join(
            line for line in chunk if not looks_like_injection(line)
        )
    if len(text) > 1600:
        text = text[:1600] + "\n..."
    return text
