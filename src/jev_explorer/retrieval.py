from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from jev_explorer.models import Artifact, RepoIndex

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "be",
    "this",
    "that",
    "where",
    "what",
    "which",
    "how",
    "does",
    "do",
    "did",
    "if",
    "we",
    "i",
    "it",
    "from",
    "by",
    "at",
    "as",
    "into",
    "about",
    "actually",
    "most",
    "likely",
    "would",
    "hit",
    "hits",
    "any",
}

FILE_HIERARCHY_THRESHOLD = 2500
FILE_SEED_LIMIT = 40
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
DOC_SUFFIXES = {".md", ".rst", ".txt"}
DOC_QUERY_TOKENS = {"documentation", "docsite", "docs", "sphinx", "readme", "changelog"}


@dataclass
class RetrievalIndex:
    df: dict[str, int] = field(default_factory=dict)
    avgdl: float = 1.0
    n_docs: int = 1
    tf: dict[str, dict[str, int]] = field(default_factory=dict)
    postings: dict[str, list[str]] = field(default_factory=dict)
    doc_len: dict[str, int] = field(default_factory=dict)
    file_ids: list[str] = field(default_factory=list)


def build_retrieval(index: RepoIndex) -> RetrievalIndex:
    tf: dict[str, dict[str, int]] = {}
    postings: dict[str, list[str]] = defaultdict(list)
    df: Counter[str] = Counter()
    doc_len: dict[str, int] = {}
    file_ids: list[str] = []
    total_len = 0
    for artifact in index.artifacts.values():
        counts = Counter(artifact_tokens(artifact))
        tf[artifact.id] = dict(counts)
        length = max(sum(counts.values()), 1)
        doc_len[artifact.id] = length
        total_len += length
        for token in counts:
            df[token] += 1
            postings[token].append(artifact.id)
        if artifact.symbol is None:
            file_ids.append(artifact.id)
    n_docs = max(len(index.artifacts), 1)
    retrieval = RetrievalIndex(
        df=dict(df),
        avgdl=total_len / n_docs if n_docs else 1.0,
        n_docs=n_docs,
        tf=tf,
        postings=dict(postings),
        doc_len=doc_len,
        file_ids=file_ids,
    )
    index.retrieval = retrieval
    return retrieval


def ensure_retrieval(index: RepoIndex) -> RetrievalIndex:
    if isinstance(index.retrieval, RetrievalIndex):
        return index.retrieval
    return build_retrieval(index)


def clean_issue_text(text: str) -> str:
    """Drop GitHub issue-template HTML comments that otherwise dominate BM25."""
    return HTML_COMMENT.sub(" ", text or "")


def seed_candidates(
    index: RepoIndex,
    question: str,
    symbols: list[str] | None = None,
    limit: int = 12,
    path_prefixes: list[str] | None = None,
) -> list[tuple[str, float]]:
    retrieval = ensure_retrieval(index)
    query_tokens = expand_tokens(tokenize(clean_issue_text(question)))
    if symbols:
        query_tokens.extend(expand_tokens(symbols))
    allowed = allowed_ids(index, path_prefixes)
    if len(index.artifacts) >= FILE_HIERARCHY_THRESHOLD:
        return hierarchical_seed(index, retrieval, query_tokens, symbols, limit, allowed)
    candidate_ids = collect_candidates(index, retrieval, query_tokens, symbols, allowed)
    return rank_ids(index, retrieval, candidate_ids, query_tokens, symbols, limit)


def hierarchical_seed(
    index: RepoIndex,
    retrieval: RetrievalIndex,
    query_tokens: list[str],
    symbols: list[str] | None,
    limit: int,
    allowed: set[str] | None,
) -> list[tuple[str, float]]:
    file_ids = [item for item in retrieval.file_ids if allowed is None or item in allowed]
    file_candidates = set(file_ids)
    for token in query_tokens:
        file_candidates.update(
            artifact_id
            for artifact_id in retrieval.postings.get(token, [])
            if index.artifacts[artifact_id].symbol is None
            and (allowed is None or artifact_id in allowed)
        )
    top_files = rank_ids(index, retrieval, file_candidates, query_tokens, symbols, FILE_SEED_LIMIT)
    nested: set[str] = set()
    for file_id, _score in top_files:
        path = index.artifacts[file_id].path
        nested.update(index.by_path.get(path, [file_id]))
    if allowed is not None:
        nested = {item for item in nested if item in allowed}
    if symbols:
        for symbol in symbols:
            nested.update(index.by_symbol.get(symbol, []))
    return rank_ids(index, retrieval, nested, query_tokens, symbols, limit)


def collect_candidates(
    index: RepoIndex,
    retrieval: RetrievalIndex,
    query_tokens: list[str],
    symbols: list[str] | None,
    allowed: set[str] | None,
) -> set[str]:
    candidate_ids: set[str] = set()
    for token in query_tokens:
        for artifact_id in retrieval.postings.get(token, []):
            if allowed is None or artifact_id in allowed:
                candidate_ids.add(artifact_id)
    if symbols:
        for symbol in symbols:
            for artifact_id in index.by_symbol.get(symbol, []):
                if allowed is None or artifact_id in allowed:
                    candidate_ids.add(artifact_id)
    if candidate_ids:
        return candidate_ids
    fallback = retrieval.file_ids if allowed is None else [item for item in retrieval.file_ids if item in allowed]
    return set(fallback)


def rank_ids(
    index: RepoIndex,
    retrieval: RetrievalIndex,
    candidate_ids: set[str],
    query_tokens: list[str],
    symbols: list[str] | None,
    limit: int,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for artifact_id in candidate_ids:
        artifact = index.artifacts.get(artifact_id)
        if artifact is None:
            continue
        score = bm25_from_retrieval(query_tokens, artifact, retrieval)
        if symbols and artifact.symbol in symbols:
            score += 4.0
        score += source_bias(query_tokens, artifact)
        if score > 0:
            ranked.append((artifact_id, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def allowed_ids(index: RepoIndex, path_prefixes: list[str] | None) -> set[str] | None:
    if not path_prefixes:
        return None
    prefixes = [item.replace("\\", "/").lstrip("./") for item in path_prefixes]
    allowed: set[str] = set()
    for artifact in index.artifacts.values():
        if any(artifact.path == prefix or artifact.path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes):
            allowed.add(artifact.id)
    return allowed


def tokenize(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
    tokens: list[str] = []
    for token in raw:
        if token in STOPWORDS or len(token) < 3:
            continue
        tokens.append(token)
        tokens.extend(part for part in split_ident(token) if part not in STOPWORDS and len(part) >= 3)
    return list(dict.fromkeys(tokens))


def expand_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(morphological_variants(token))
    return list(dict.fromkeys(expanded))


def morphological_variants(token: str) -> list[str]:
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        stem = token[:-3] + "y"
        variants.add(stem)
        variants.update(morphological_variants(stem) if stem != token else [])
    if token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
        variants.add(token[:-1])
    elif token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    if token.endswith("ing") and len(token) > 5:
        variants.add(token[:-3])
        variants.add(token[:-3] + "e")
    if token.endswith("ed") and len(token) > 4:
        variants.add(token[:-2])
        variants.add(token[:-1])
    if token.endswith("y") and len(token) > 3:
        stem = token[:-1]
        variants.add(stem + "e")
        variants.add(stem + "es")
        variants.add(stem + "ed")
        variants.add(stem)
    if token.endswith("en") and len(token) > 4:
        variants.add(token[:-2])
        variants.add(token[:-1])
    return [item for item in variants if item and item not in STOPWORDS and len(item) >= 3]


def split_ident(token: str) -> list[str]:
    parts = re.split(r"[_]+", token)
    camel = re.findall(r"[a-z]+|[A-Z][a-z]*", token)
    return [part.lower() for part in parts + camel if part]


def artifact_tokens(artifact: Artifact) -> list[str]:
    blob = " ".join(
        filter(
            None,
            [artifact.path, artifact.symbol or "", artifact.signature, artifact.excerpt, artifact.docstring or ""],
        )
    )
    return tokenize(blob)


def bm25_from_retrieval(
    query_tokens: list[str],
    artifact: Artifact,
    retrieval: RetrievalIndex,
    k1: float = 1.4,
    b: float = 0.75,
) -> float:
    tf = retrieval.tf.get(artifact.id, {})
    dl = max(retrieval.doc_len.get(artifact.id, 1), 1)
    score = 0.0
    n_docs = retrieval.n_docs
    avgdl = retrieval.avgdl
    df = retrieval.df
    for token in query_tokens:
        freq = tf.get(token, 0)
        if freq == 0:
            continue
        idf = math.log(1 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
        denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1.0))
        score += idf * (freq * (k1 + 1)) / denom
    return score + name_boost(query_tokens, artifact)


def source_bias(query_tokens: list[str], artifact: Artifact) -> float:
    path = artifact.path.lower()
    suffix = path[path.rfind(".") :] if "." in path.rsplit("/", 1)[-1] else ""
    if "/examples/" in path or path.startswith("examples/"):
        return -2.5
    if suffix in DOC_SUFFIXES and not any(token in DOC_QUERY_TOKENS for token in query_tokens):
        return -3.0
    if artifact.kind in {"function", "class", "module"}:
        return 0.6
    return 0.0


def compact_ident(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def name_boost(query_tokens: list[str], artifact: Artifact) -> float:
    path_boost = 0.4 * sum(1 for token in query_tokens if token in artifact.path.lower())
    symbol_boost = 0.6 * sum(1 for token in query_tokens if artifact.symbol and token in artifact.symbol.lower())
    filename = artifact.path.rsplit("/", 1)[-1].lower()
    stem = filename.rsplit(".", 1)[0]
    name = 0.0
    stem_parts = set(stem.split("_")) | set(tokenize(stem))
    compact_stem = compact_ident(stem)
    compact_symbol = compact_ident(artifact.symbol or "")
    for token in query_tokens:
        if token == stem or token in stem_parts:
            name += 2.2
            continue
        compact_token = compact_ident(token)
        if len(compact_token) >= 4 and compact_token == compact_stem:
            name += 2.2
        elif len(compact_token) >= 4 and compact_symbol and compact_token == compact_symbol:
            name += 1.8
    return path_boost + symbol_boost + name


def bm25(
    query_tokens: list[str],
    artifact: Artifact,
    df: dict[str, int],
    n_docs: int,
    avgdl: float,
    k1: float = 1.4,
    b: float = 0.75,
) -> float:
    tf = Counter(artifact_tokens(artifact))
    retrieval = RetrievalIndex(df=df, avgdl=avgdl, n_docs=n_docs, tf={artifact.id: dict(tf)}, doc_len={artifact.id: max(sum(tf.values()), 1)})
    return bm25_from_retrieval(query_tokens, artifact, retrieval, k1=k1, b=b)
