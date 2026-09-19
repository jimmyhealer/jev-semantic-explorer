from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitIgnore:
    """Root .gitignore matcher. Good enough for source walks, not a full git implementation."""

    rules: list[tuple[bool, str]]

    @classmethod
    def from_root(cls, root: Path) -> GitIgnore:
        rules: list[tuple[bool, str]] = []
        path = Path(root) / ".gitignore"
        if not path.is_file():
            return cls(rules)
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:]
            rules.append((negated, line.replace("\\", "/").rstrip("/")))
        return cls(rules)

    def matches(self, rel: str, is_dir: bool = False) -> bool:
        rel = rel.replace("\\", "/").lstrip("./")
        if not rel:
            return False
        ignored = False
        for negated, pattern in self.rules:
            if _rule_matches(pattern, rel, is_dir):
                ignored = not negated
        return ignored


def _rule_matches(pattern: str, rel: str, is_dir: bool) -> bool:
    if pattern.endswith("/"):
        if not is_dir:
            return False
        pattern = pattern.rstrip("/")
    if pattern.startswith("/"):
        pattern = pattern.lstrip("/")
        return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel.split("/", 1)[0], pattern)
    name = rel.rsplit("/", 1)[-1]
    if "/" in pattern:
        return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern.rstrip("/**"))
    if fnmatch.fnmatch(name, pattern):
        return True
    return any(fnmatch.fnmatch(part, pattern) for part in rel.split("/"))
