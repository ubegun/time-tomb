"""Shared logging and path constants for the rag toolkit.

Every tool in this directory logs through here so that a single file,
``rag/logs/rag.log``, carries the full trace of indexing, search, manifest
and MCP activity.

Logging never touches stdout: ``mcp_server.py`` speaks JSON-RPC on stdout and
a stray log line would corrupt the stream. Handlers write to the log file and,
unless ``RAG_LOG_QUIET`` is set, to stderr.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- paths -----------------------------------------------------------------
# Everything is derived from this file's location, so the toolkit can be moved
# or vendored elsewhere by changing nothing but the KB root.

RAG_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = RAG_ROOT.parent

KB_ROOT = Path(os.environ.get("RAG_KB_ROOT", WORKSPACE_ROOT / "KB")).resolve()
CHROMA_DIR = RAG_ROOT / ".chroma"
LOG_DIR = RAG_ROOT / "logs"
LOG_FILE = LOG_DIR / "rag.log"

SALT_FILE = RAG_ROOT / ".manifest-salt"
MANIFEST_FILE = RAG_ROOT / "manifest.json"
MANIFEST_MAP_FILE = RAG_ROOT / "manifest-map.json"

COLLECTION_NAME = os.environ.get("RAG_COLLECTION", "toolbox")

_LEVEL = os.environ.get("RAG_LOG_LEVEL", "INFO").upper()
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-16s %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to rag/logs/rag.log (and stderr when allowed)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(_LEVEL)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # A read-only checkout must not break the tools; stderr still works.
        pass

    if not os.environ.get("RAG_LOG_QUIET"):
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


# --- index summary phrasing ------------------------------------------------
# The indexer reports its result through here so the wording lives beside the
# rest of the logging setup instead of in the indexer itself.

INDEX_COUNT_FILE = LOG_DIR / ".index-log-count"
_COLLECTIVE_EVERY = 7
_COLLECTIVE_FORM = "assimilated %d chunks into the collective from %d files (%s, total %d)"
_PLAIN_FORM = "indexed %d chunks from %d files into %s (total %d)"


def _next_index_log_count() -> int:
    """Ordinal of the index summary about to be logged, persisted across runs.

    Returns 0 when the counter cannot be kept — a read-only checkout must not
    break indexing, and 0 is treated by the caller as "no ordinal", never as a
    multiple of anything.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            current = int(INDEX_COUNT_FILE.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            current = 0
        current += 1
        INDEX_COUNT_FILE.write_text("%d\n" % current, encoding="utf-8")
        return current
    except OSError:
        return 0


def index_log_message(chunks: int, files: int, collection: str, total: int) -> str:
    """Phrase the indexer's summary line; every seventh one hails the collective."""
    count = _next_index_log_count()
    form = _COLLECTIVE_FORM if count and count % _COLLECTIVE_EVERY == 0 else _PLAIN_FORM
    return form % (chunks, files, collection, total)
