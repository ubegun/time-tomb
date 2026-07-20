#!/usr/bin/env python
"""Chunk the markdown knowledge base into a local Chroma collection.

Chunking is heading-aware: a markdown file is split at ATX headings, and any
section longer than ``MAX_CHARS`` is further split on paragraph boundaries with
a small overlap so that a retrieved chunk keeps enough surrounding context.

The chunker is also imported by ``manifest.py`` — the manifest hashes exactly
the chunks that were indexed, so the published skeleton and the local vector
store always describe the same body.

Chunk geometry is configurable through a *profile* (``profiles/<name>.json``)
so that different cuts of the same body can be compared. Configuration is
strictly opt-in: with no profile named, every function below behaves exactly
as it did before profiles existed, down to the byte, and therefore produces
the same manifest.

Usage:
    python index_toolbox.py            # incremental re-index
    python index_toolbox.py --reset    # drop the collection first
    python index_toolbox.py --stats    # report without writing
    python index_toolbox.py --profile chunk-small --reset
"""

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from raglog import (
    CHROMA_DIR,
    COLLECTION_NAME,
    KB_ROOT,
    RAG_ROOT,
    get_logger,
    index_log_message,
)

log = get_logger("index")

MAX_CHARS = 1200
OVERLAP_CHARS = 150
MIN_CHARS = 60  # below this a chunk carries no retrievable signal

PROFILE_DIR = RAG_ROOT / "profiles"
PROFILE_ENV = "RAG_PROFILE"
SUPPORTED_KINDS = ("local",)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class ChunkConfig:
    """How a document is cut into retrievable units.

    The field defaults are the module constants above, i.e. the shipped
    behaviour. ``ChunkConfig()`` and "no profile selected" are the same thing.
    """

    max_chars: int = MAX_CHARS
    overlap_chars: int = OVERLAP_CHARS
    min_chars: int = MIN_CHARS
    strategy: str = "heading-aware"


DEFAULT_CONFIG = ChunkConfig()


# --- profiles --------------------------------------------------------------


class ProfileError(RuntimeError):
    """Raised when a profile is missing, malformed or not implemented here."""


def profile_path(name: str) -> Path:
    return PROFILE_DIR / ("%s.json" % name)


def available_profiles() -> List[str]:
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def load_profile(name: str) -> dict:
    """Read and validate ``profiles/<name>.json``."""
    path = profile_path(name)
    if not path.is_file():
        raise ProfileError(
            "no profile %r in %s (have: %s)"
            % (name, PROFILE_DIR, ", ".join(available_profiles()) or "none")
        )
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError("cannot read profile %r: %s" % (name, exc)) from exc

    if not isinstance(profile, dict):
        raise ProfileError("profile %r is not a JSON object" % name)
    if profile.get("name") != name:
        raise ProfileError(
            "profile %r declares name %r - filename and name must agree"
            % (name, profile.get("name"))
        )

    kind = profile.get("kind")
    if kind not in SUPPORTED_KINDS:
        # A remote contender is a planned shape, not a supported one. Refuse
        # loudly rather than silently falling back to the local path.
        raise ProfileError(
            "profile %r has kind %r; this toolkit implements %s only"
            % (name, kind, "/".join(SUPPORTED_KINDS))
        )
    return profile


def config_from_profile(profile: dict) -> ChunkConfig:
    """Build a ChunkConfig from a profile's ``chunking`` block."""
    block = profile.get("chunking") or {}
    strategy = block.get("strategy", DEFAULT_CONFIG.strategy)
    if strategy != DEFAULT_CONFIG.strategy:
        raise ProfileError(
            "chunking strategy %r is not implemented (only %r)"
            % (strategy, DEFAULT_CONFIG.strategy)
        )
    config = ChunkConfig(
        max_chars=int(block.get("max_chars", MAX_CHARS)),
        overlap_chars=int(block.get("overlap_chars", OVERLAP_CHARS)),
        min_chars=int(block.get("min_chars", MIN_CHARS)),
        strategy=strategy,
    )
    if config.max_chars < 1:
        raise ProfileError("max_chars must be positive")
    if config.overlap_chars < 0 or config.overlap_chars >= config.max_chars:
        raise ProfileError("overlap_chars must be in [0, max_chars)")
    if config.min_chars < 0:
        raise ProfileError("min_chars must not be negative")
    return config


def resolve_config(config: Optional[ChunkConfig] = None) -> ChunkConfig:
    """Pick the chunk configuration in force.

    Precedence: an explicit argument, then ``$RAG_PROFILE``, then the shipped
    defaults. The last case reads nothing from disk, so the default path stays
    exactly what it was.
    """
    if config is not None:
        return config
    name = os.environ.get(PROFILE_ENV, "").strip()
    if not name:
        return DEFAULT_CONFIG
    return config_from_profile(load_profile(name))


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of the knowledge base."""

    source: str  # KB-relative posix path
    index: int  # position within the source file
    heading: str  # nearest enclosing heading ("" at file top)
    text: str

    @property
    def uid(self) -> str:
        """Stable id: same (file, position) upserts in place across runs."""
        return "%s#%d" % (self.source, self.index)

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def _split_long(body: str, config: ChunkConfig = DEFAULT_CONFIG) -> List[str]:
    """Split an over-long section on paragraph breaks, keeping an overlap."""
    max_chars = config.max_chars
    overlap_chars = config.overlap_chars
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else current + "\n\n" + paragraph
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        parts.append(current)
        tail = current[-overlap_chars:] if overlap_chars else ""
        current = (tail + "\n\n" + paragraph) if tail else paragraph

    if current:
        parts.append(current)

    # A single paragraph can still exceed the budget; hard-wrap those.
    wrapped: List[str] = []
    for part in parts:
        while len(part) > max_chars * 2:
            wrapped.append(part[:max_chars])
            part = part[max_chars - overlap_chars:]
        wrapped.append(part)
    return wrapped


def chunk_markdown(
    text: str, source: str, config: Optional[ChunkConfig] = None
) -> List[Chunk]:
    """Split one markdown document into heading-scoped chunks."""
    config = resolve_config(config)
    sections: List[tuple] = []  # (heading, body lines)
    heading = ""
    buffer: List[str] = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if buffer:
                sections.append((heading, "\n".join(buffer)))
                buffer = []
            heading = match.group(2)
            continue
        buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer)))

    chunks: List[Chunk] = []
    for section_heading, body in sections:
        body = body.strip()
        if not body:
            continue
        pieces = [body] if len(body) <= config.max_chars else _split_long(body, config)
        for piece in pieces:
            piece = piece.strip()
            if len(piece) < config.min_chars:
                continue
            # The heading is prepended to the embedded text so that a chunk
            # retrieved out of context still states what it is about.
            embedded = ("%s\n\n%s" % (section_heading, piece)) if section_heading else piece
            chunks.append(
                Chunk(
                    source=source,
                    index=len(chunks),
                    heading=section_heading,
                    text=embedded,
                )
            )
    return chunks


def iter_kb_files(root: Optional[Path] = None) -> List[Path]:
    """Markdown notes of the knowledge base, in stable order."""
    root = root or KB_ROOT
    if not root.is_dir():
        return []
    files = [
        p
        for p in sorted(root.rglob("*.md"))
        if not p.name.startswith(".") and "CLAUDE.md" not in p.name
    ]
    return files


def collect_chunks(
    root: Optional[Path] = None, config: Optional[ChunkConfig] = None
) -> List[Chunk]:
    """Every chunk of the knowledge base, deterministically ordered."""
    root = root or KB_ROOT
    config = resolve_config(config)
    chunks: List[Chunk] = []
    for path in iter_kb_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("skipping %s: %s", rel, exc)
            continue
        chunks.extend(chunk_markdown(text, rel, config))
    return chunks


def get_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _embedding_function():
    """Local ONNX MiniLM — no network calls once the model is cached."""
    from chromadb.utils import embedding_functions

    try:
        return embedding_functions.ONNXMiniLM_L6_V2()
    except Exception:  # pragma: no cover - depends on chromadb build
        return embedding_functions.DefaultEmbeddingFunction()


def get_collection(create: bool = True, name: Optional[str] = None):
    client = get_client()
    name = name or COLLECTION_NAME
    if create:
        return client.get_or_create_collection(
            name=name,
            embedding_function=_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=name, embedding_function=_embedding_function())


def reindex(
    reset: bool = False,
    root: Optional[Path] = None,
    collection_name: Optional[str] = None,
    config: Optional[ChunkConfig] = None,
) -> dict:
    """Rebuild the collection from the knowledge base. Returns a summary."""
    root = root or KB_ROOT
    name = collection_name or COLLECTION_NAME
    config = resolve_config(config)
    client = get_client()

    if reset:
        try:
            client.delete_collection(name)
            log.info("dropped collection %s", name)
        except Exception:
            pass  # nothing to drop on a first run

    collection = get_collection(name=name)
    chunks = collect_chunks(root, config)
    sources = sorted({c.source for c in chunks})

    if not chunks:
        log.warning("no chunks found under %s", root)
        return {"files": 0, "chunks": 0, "collection": name}

    # Drop each touched file's old chunks so that shrinking a note does not
    # leave orphans behind, then write the current ones.
    for source in sources:
        try:
            collection.delete(where={"source": source})
        except Exception as exc:
            log.debug("no prior chunks for %s (%s)", source, exc)

    collection.upsert(
        ids=[c.uid for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "source": c.source,
                "heading": c.heading,
                "chunk_index": c.index,
                "sha": c.sha,
            }
            for c in chunks
        ],
    )

    summary = {
        "files": len(sources),
        "chunks": len(chunks),
        "collection": name,
        "count": collection.count(),
    }
    log.info(
        "%s",
        index_log_message(
            summary["chunks"],
            summary["files"],
            name,
            summary["count"],
        ),
    )
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument("--stats", action="store_true", help="report chunking, write nothing")
    parser.add_argument("--root", type=Path, default=None, help="knowledge base root")
    parser.add_argument(
        "--profile",
        default=None,
        help="profiles/<name>.json to take chunking and collection from",
    )
    parser.add_argument(
        "--collection", default=None, help="override the target collection name"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.root.resolve() if args.root else KB_ROOT

    collection_name = args.collection
    config: Optional[ChunkConfig] = None
    profile_label = "(defaults)"
    if args.profile:
        try:
            profile = load_profile(args.profile)
        except ProfileError as exc:
            print("profile error: %s" % exc)
            return 2
        config = config_from_profile(profile)
        collection_name = collection_name or profile.get("collection")
        profile_label = args.profile
    else:
        config = resolve_config(None)
        if os.environ.get(PROFILE_ENV, "").strip():
            profile_label = "$%s=%s" % (PROFILE_ENV, os.environ[PROFILE_ENV])

    if args.stats:
        chunks = collect_chunks(root, config)
        by_source: dict = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, 0)
            by_source[chunk.source] += 1
        print("root:   %s" % root)
        print("profile: %s" % profile_label)
        print(
            "cutter: max=%d overlap=%d min=%d strategy=%s"
            % (config.max_chars, config.overlap_chars, config.min_chars, config.strategy)
        )
        print("files:  %d" % len(by_source))
        print("chunks: %d" % len(chunks))
        for source, count in sorted(by_source.items()):
            print("  %-60s %3d" % (source, count))
        return 0

    summary = reindex(
        reset=args.reset, root=root, collection_name=collection_name, config=config
    )
    print(
        "indexed %d chunks from %d files -> collection '%s' (%d docs)"
        % (summary["chunks"], summary["files"], summary["collection"], summary.get("count", 0))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
