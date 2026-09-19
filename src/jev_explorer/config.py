from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def skill_path() -> Path:
    return repo_root() / "skills" / "jev-explorer" / "SKILL.md"


def load_skill_text() -> str:
    path = skill_path()
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def load_env() -> None:
    candidates = [Path.cwd() / ".env", repo_root() / ".env"]
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)


def api_key() -> str:
    load_env()
    key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing API key. Set JEV_API_KEY (or TYPESAFE_API_KEY) in .env."
        )
    return key
