from __future__ import annotations

import os
from typing import Any

from jev_explorer.config import load_env
from jev_explorer.investigate import codebase_investigate
from jev_explorer.models import Budget, EvidencePolicy, InvestigateRequest, Scope


def default_repo() -> str:
    return os.environ.get("JEV_EXPLORER_REPO") or os.getcwd()


def default_index_path() -> str | None:
    return os.environ.get("JEV_EXPLORER_INDEX")


def investigate_tool(
    question: str,
    repo: str | None = None,
    paths: list[str] | None = None,
    symbols: list[str] | None = None,
    max_nodes: int = 18,
    max_depth: int = 3,
    max_branch: int = 4,
    max_jev_tokens: int = 80_000,
    require_tests: bool = False,
    require_docs: bool = False,
    allow_generated: bool = False,
    use_jev: bool = True,
    index_path: str | None = None,
) -> dict[str, Any]:
    request = InvestigateRequest(
        question=question,
        repo=repo or default_repo(),
        scope=Scope(paths=paths or [], symbols=symbols or []),
        budget=Budget(
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_branch=max_branch,
            max_jev_tokens=max_jev_tokens,
        ),
        evidence_policy=EvidencePolicy(
            require_tests=require_tests,
            require_docs=require_docs,
            allow_generated=allow_generated,
        ),
        use_jev=use_jev,
        index_path=index_path or default_index_path(),
    )
    return codebase_investigate(request).to_dict()


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer(
        "jev-explorer",
        instructions=(
            "Read-only jevex for coding agents. Call codebase_investigate with an "
            "engineering question before grepping the repo. It returns an evidence "
            "packet plus ranked regions. Do not ask it to edit code. Do not call "
            "jev_evaluate."
        ),
    )

    @mcp.tool()
    def codebase_investigate(  # noqa: F811
        question: str,
        repo: str | None = None,
        paths: list[str] | None = None,
        symbols: list[str] | None = None,
        max_nodes: int = 18,
        max_depth: int = 3,
        max_branch: int = 4,
        max_jev_tokens: int = 80_000,
        require_tests: bool = False,
        require_docs: bool = False,
        allow_generated: bool = False,
        use_jev: bool = True,
        index_path: str | None = None,
    ) -> dict[str, Any]:
        """Investigate a repository question and return a traceable evidence packet.

        Use this instead of grep/read when you need the source of truth, related
        tests, docs, and impact paths. The packet includes ranked code regions.
        """
        return investigate_tool(
            question=question,
            repo=repo,
            paths=paths,
            symbols=symbols,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_branch=max_branch,
            max_jev_tokens=max_jev_tokens,
            require_tests=require_tests,
            require_docs=require_docs,
            allow_generated=allow_generated,
            use_jev=use_jev,
            index_path=index_path,
        )

    return mcp


def main() -> None:
    load_env()
    build_server().run()


if __name__ == "__main__":
    main()
