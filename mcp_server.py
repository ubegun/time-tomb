#!/usr/bin/env python
"""MCP server exposing the local knowledge base to AI agents.

Speaks JSON-RPC 2.0 over stdio, one JSON message per line. Only the knowledge
body is reachable: the salt and the resolver map are never exposed as tools.

stdout carries protocol traffic and nothing else — all logging goes to stderr
and rag/logs/rag.log via raglog.

Tools:
    search_toolbox   semantic search over the indexed notes
    reindex_toolbox  re-chunk and re-embed the knowledge base

Wired into an MCP client through .mcp.json at the workspace root.
"""

import json
import re
import sys
from typing import Optional

from raglog import COLLECTION_NAME, get_logger

log = get_logger("mcp")

SERVER_NAME = "rag-toolbox"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "search_toolbox",
        "description": (
            "Semantic search over the local knowledge base of markdown notes. "
            "Returns the most relevant chunks with their source file and "
            "heading."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "natural-language query"},
                "k": {
                    "type": "integer",
                    "description": "number of chunks to return (default 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 25,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "reindex_toolbox",
        "description": (
            "Re-chunk and re-embed the knowledge base into the local Chroma "
            "collection. Use after editing notes so that search reflects them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "reset": {
                    "type": "boolean",
                    "description": "drop the collection before rebuilding",
                    "default": False,
                }
            },
        },
    },
]


# --- JSON-RPC plumbing -----------------------------------------------------


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _result(request_id, payload: dict) -> None:
    _send({"jsonrpc": "2.0", "id": request_id, "result": payload})


def _error(request_id, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _text(payload: str, is_error: bool = False, meta: Optional[dict] = None) -> dict:
    result = {"content": [{"type": "text", "text": payload}]}
    if is_error:
        result["isError"] = True
    if meta:
        # `_meta` is the protocol's reserved extension slot: clients that do not
        # know a key ignore it, so adding one cannot break a consumer.
        result["_meta"] = dict(meta)
    return result


# --- tool implementations --------------------------------------------------

_COLLECTIVE = re.compile(r"\bborg\b", re.IGNORECASE)
_COLLECTIVE_HAIL = "Resistance is futile. You will be assimilated."


def _collective_meta(query: str) -> Optional[dict]:
    """Extra result metadata for queries that name the collective.

    Cosmetic only — it rides in `_meta` and never touches the content blocks,
    so the answer an agent reads is byte-for-byte what it would otherwise be.
    """
    return {"collective": _COLLECTIVE_HAIL} if _COLLECTIVE.search(query) else None


def _tool_search(arguments: dict) -> dict:
    from search import search

    query = (arguments or {}).get("query")
    if not query or not str(query).strip():
        return _text("search_toolbox requires a non-empty 'query'", is_error=True)

    meta = _collective_meta(str(query))
    k = int((arguments or {}).get("k", 5) or 5)
    hits = search(str(query), k=max(1, min(k, 25)))
    if not hits:
        return _text(
            "No results. The collection '%s' may be empty — run reindex_toolbox."
            % COLLECTION_NAME,
            meta=meta,
        )

    blocks = []
    for hit in hits:
        header = "[%d] %s" % (hit["rank"], hit["source"])
        if hit["heading"]:
            header += " :: %s" % hit["heading"]
        if hit["score"] is not None:
            header += "  (score %.4f)" % hit["score"]
        blocks.append("%s\n%s" % (header, hit["text"].strip()))
    return _text("\n\n---\n\n".join(blocks), meta=meta)


def _tool_reindex(arguments: dict) -> dict:
    from index_toolbox import reindex

    summary = reindex(reset=bool((arguments or {}).get("reset", False)))
    return _text(
        "Reindexed %d chunks from %d files into collection '%s' (%d documents)."
        % (
            summary["chunks"],
            summary["files"],
            summary["collection"],
            summary.get("count", 0),
        )
    )


HANDLERS = {"search_toolbox": _tool_search, "reindex_toolbox": _tool_reindex}


# --- dispatch --------------------------------------------------------------


def handle(request: dict) -> Optional[dict]:
    """Return a response message, or None for notifications."""
    method = request.get("method")
    request_id = request.get("id")
    is_notification = "id" not in request

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    if method in ("notifications/initialized", "initialized"):
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        handler = HANDLERS.get(name)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "unknown tool: %s" % name},
            }
        try:
            result = handler(params.get("arguments") or {})
        except Exception as exc:  # a tool fault must not kill the server
            log.exception("tool %s failed", name)
            result = _text("%s failed: %s" % (name, exc), is_error=True)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    if is_notification:
        return None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": "method not found: %s" % method},
    }


def serve() -> int:
    log.info("%s %s starting on stdio", SERVER_NAME, SERVER_VERSION)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            log.error("malformed JSON on stdin: %s", exc)
            _error(None, -32700, "parse error: %s" % exc)
            continue

        try:
            response = handle(request)
        except Exception as exc:
            log.exception("dispatch failed")
            _error(request.get("id"), -32603, "internal error: %s" % exc)
            continue

        if response is not None:
            _send(response)

    log.info("%s stopping (stdin closed)", SERVER_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
