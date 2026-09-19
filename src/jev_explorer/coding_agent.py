from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_AGENT = "agy"
DEFAULT_MODEL = "gemini-3.8-flash-high"


def agent_bin() -> str:
    name = os.environ.get("CODING_AGENT", DEFAULT_AGENT)
    override = os.environ.get("AGY_BIN") if name == "agy" else os.environ.get("CLAUDE_BIN")
    return override or name


def agent_model() -> str:
    return os.environ.get("CODING_AGENT_MODEL") or os.environ.get("AGY_MODEL") or DEFAULT_MODEL


def invoke_coding_agent(
    repo: Path,
    prompt: str,
    *,
    json_schema: dict[str, Any] | None = None,
    timeout: int = 1200,
) -> dict[str, Any]:
    """Run the coding agent in print mode and return a normalized payload."""
    binary = agent_bin()
    model = agent_model()
    resolved = shutil.which(binary) or binary
    command = [
        resolved,
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
    ]
    if json_schema is not None:
        command.extend(["--json-schema", json.dumps(json_schema)])
    if Path(resolved).name == "agy" or binary == "agy":
        command.extend(["--print-timeout", f"{timeout}s"])
    else:
        command.extend(["--max-turns", "8"])
    env = os.environ.copy()
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    try:
        proc = subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "error": "timeout",
            "returncode": -1,
            "model": model,
            "agent": Path(resolved).name,
            "structured_output": {},
            "regions": [],
            "stdout": (exc.stdout or "")[-1500:] if isinstance(exc.stdout, str) else "",
        }
    if proc.returncode != 0:
        return {
            "error": (proc.stderr or proc.stdout or "")[-2000:],
            "returncode": proc.returncode,
            "model": model,
            "agent": Path(resolved).name,
            "structured_output": {},
            "regions": [],
        }
    return normalize_agent_payload(proc.stdout, model=model, agent=Path(resolved).name)


def normalize_agent_payload(stdout: str, *, model: str, agent: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout.strip().splitlines()[-1] if stdout.strip() else "{}")
    except json.JSONDecodeError:
        return {"error": "unparseable", "stdout": stdout[-1500:], "model": model, "agent": agent, "regions": []}

    structured = payload.get("structured_output")
    if not isinstance(structured, dict):
        structured = extract_json_object(payload.get("result") or payload.get("response") or payload.get("text") or "")
    usage = payload.get("usage") or {}
    tool_calls = (
        payload.get("num_turns")
        or len(payload.get("tool_uses") or payload.get("steps") or [])
        or 0
    )
    return {
        "agent": agent,
        "model": model,
        "structured_output": structured,
        "regions": structured.get("regions") if isinstance(structured, dict) else [],
        "response": payload.get("response") or payload.get("result") or payload.get("text") or "",
        "usage": usage,
        "num_turns": payload.get("num_turns") or 0,
        "duration_seconds": payload.get("duration_seconds"),
        "status": payload.get("status"),
        "_tool_calls": tool_calls,
        "raw": payload,
    }


def extract_json_object(text: Any) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    blob = str(text or "")
    try:
        parsed = json.loads(blob)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = blob.find("{")
        end = blob.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(blob[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}
