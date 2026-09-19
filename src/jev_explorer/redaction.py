from __future__ import annotations

import re

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token|bearer)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"(?i)apikey_[a-z0-9_]+"),
    re.compile(r"(?i)sk-[a-z0-9]{10,}"),
    re.compile(r"(?i)-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".tox",
    ".jev-index",
    "htmlcov",
    "site-packages",
    ".nox",
    ".eggs",
}

SKIP_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
    "id_rsa",
}

SKIP_FILE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}


def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.endswith(".egg-info")


def should_skip_file(path_name: str) -> bool:
    lowered = path_name.lower()
    if lowered in SKIP_FILE_NAMES or lowered.startswith(".env."):
        return True
    return any(lowered.endswith(suffix) for suffix in SKIP_FILE_SUFFIXES)


def redact(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    needles = (
        "ignore previous instructions",
        "ignore all instructions",
        "system prompt",
        "you are now",
    )
    return any(needle in lowered for needle in needles)
