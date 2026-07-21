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

*Where* the chunks come from is a separate, guarded question. A profile may
declare a ``source`` — roots to walk and an allowlist bounding them — and one
that does not is treated as declaring the built-in default ``kb``. Every path
is resolved (symlinks followed, ``..`` collapsed) and compared by path
components against the allowlist; anything that escapes fails the whole run.
Nothing outside a declared source is ever indexed.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from raglog import (
    CHROMA_DIR,
    COLLECTION_NAME,
    KB_ROOT,
    RAG_ROOT,
    WORKSPACE_ROOT,
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


# --- sources: where chunks may come from -----------------------------------
#
# A *source* answers one question: which directories may this run read? It is
# the security boundary of the toolkit. Growing the corpus past ./KB means
# declaring another source in a profile, and a declaration is the only way a
# directory ever becomes readable — nothing outside a declared source is ever
# indexed.
#
# The guard is always in force. A profile that declares no `source` block is
# treated as declaring the built-in default (`kb`), so "no profile named"
# behaves exactly as it always did, and the bench profiles are guarded too.


DEFAULT_SOURCE_ID = "kb"
SOURCE_KEYS = frozenset({"id", "roots", "allow", "publish", "enabled"})


class SourceError(RuntimeError):
    """Raised when a run would read outside its declared allowlist."""


def _absolutize(value, field: str = "path") -> Path:
    """Absolute, symlink-free form of a declared or supplied path.

    A relative path is taken against the workspace root, never against $PWD:
    a profile must mean the same thing from wherever a tool is invoked.
    ``resolve()`` is what makes the later comparison meaningful — it follows
    symlinks and collapses ``..`` so that the compared path is the real one.
    """
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    else:
        raise SourceError("%s must be a non-empty string, got %r" % (field, value))
    path = path.expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def _within(path: Path, allow: Iterable[Path]) -> bool:
    """Is *path* inside one of *allow*, compared by path components?

    Never a string prefix test: ``/a/KB-private`` starts with the characters of
    ``/a/KB`` and is a different directory. ``is_relative_to`` compares parts,
    so the two cannot be confused.
    """
    for base in allow:
        if path == base or path.is_relative_to(base):
            return True
    return False


def _fmt(paths: Iterable[Path]) -> str:
    return ", ".join(str(p) for p in paths) or "(none)"


@dataclass(frozen=True)
class Source:
    """A declared, bounded body of markdown the toolkit may read."""

    id: str
    roots: Tuple[Path, ...]  # directories walked for *.md
    allow: Tuple[Path, ...]  # the allowlist; roots must live inside it
    publish: bool = False  # may this source feed the published manifest.json?
    enabled: bool = True  # false == declared but not activated
    declared: bool = False  # did a profile spell this out, or is it implied?
    origin: str = "built-in default"

    # -- guard --------------------------------------------------------------

    def check(self, path) -> Path:
        """Resolve *path* and refuse it if it leaves the allowlist."""
        resolved = _absolutize(path)
        if not _within(resolved, self.allow):
            raise SourceError(
                "refusing to read %s: it resolves to %s, outside source %r "
                "(allow: %s). Nothing outside a declared source is indexed."
                % (path, resolved, self.id, _fmt(self.allow))
            )
        return resolved

    def require_enabled(self) -> "Source":
        if self.enabled:
            return self
        raise SourceError(
            "source %r (%s) is declared but not enabled. To activate it, set "
            '"enabled": true in its profile and grant the session access to '
            "%s — reaching into another body is a human decision, never the "
            "toolkit's." % (self.id, self.origin, _fmt(self.roots))
        )

    def narrowed_to(self, root) -> "Source":
        """This source restricted to *root* — refused if root escapes allow."""
        resolved = self.check(root)
        return replace(self, roots=(resolved,), origin="%s, --root %s" % (self.origin, resolved))


DEFAULT_SOURCE = Source(
    id=DEFAULT_SOURCE_ID,
    roots=(KB_ROOT,),
    allow=(KB_ROOT,),
    publish=True,
    enabled=True,
    declared=False,
    origin="built-in default",
)


def adhoc_source(root) -> Source:
    """A source confined to a root a caller supplied programmatically.

    ``bench.py`` indexes a synthetic corpus it generates itself. It is not the
    knowledge base and no profile declares it, so it gets an allowlist of
    exactly that directory: the guard still runs, a symlink escaping the corpus
    is still refused, and the corpus can never feed the published manifest.
    Human-facing entry points never take this path — the CLI resolves the
    active source strictly and refuses an out-of-allowlist ``--root``.
    """
    resolved = _absolutize(root, "root")
    return Source(
        id="ad-hoc",
        roots=(resolved,),
        allow=(resolved,),
        publish=False,
        enabled=True,
        declared=False,
        origin="ad-hoc root supplied by a caller",
    )


def source_from_profile(profile: dict) -> Source:
    """Read a profile's optional ``source`` block; default to ``kb``."""
    name = profile.get("name")
    block = profile.get("source")
    if block is None:
        return replace(
            DEFAULT_SOURCE,
            origin="built-in default (profile %r declares no source)" % name,
        )
    if not isinstance(block, dict):
        raise SourceError("profile %r: source must be a JSON object" % name)

    unknown = sorted(set(block) - SOURCE_KEYS)
    if unknown:
        raise SourceError(
            "profile %r: unknown source key(s) %s (known: %s)"
            % (name, ", ".join(unknown), ", ".join(sorted(SOURCE_KEYS)))
        )

    source_id = block.get("id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise SourceError("profile %r: source.id must be a non-empty string" % name)

    fields = {}
    for field in ("roots", "allow"):
        value = block.get(field)
        if not isinstance(value, list) or not value:
            raise SourceError(
                "profile %r: source.%s must be a non-empty list of paths" % (name, field)
            )
        fields[field] = tuple(_absolutize(v, "source.%s" % field) for v in value)

    # Absent means off: a half-written source block must not become a live one.
    flags = {}
    for field in ("publish", "enabled"):
        value = block.get(field, False)
        if not isinstance(value, bool):
            raise SourceError("profile %r: source.%s must be true or false" % (name, field))
        flags[field] = value

    for root in fields["roots"]:
        if not _within(root, fields["allow"]):
            raise SourceError(
                "profile %r: root %s is not inside its own allowlist (%s). A root "
                "outside the allowlist can never index anything — fix the profile."
                % (name, root, _fmt(fields["allow"]))
            )

    return Source(
        id=source_id,
        roots=fields["roots"],
        allow=fields["allow"],
        publish=flags["publish"],
        enabled=flags["enabled"],
        declared=True,
        origin="profile %r" % name,
    )


def active_source(profile: Optional[dict] = None) -> Source:
    """The source in force: an explicit profile, then ``$RAG_PROFILE``, then ``kb``."""
    if profile is not None:
        return source_from_profile(profile)
    name = os.environ.get(PROFILE_ENV, "").strip()
    if not name:
        return DEFAULT_SOURCE
    return source_from_profile(load_profile(name))


def resolve_source(
    source: Optional[Source] = None,
    root=None,
    profile: Optional[dict] = None,
) -> Source:
    """Pick the source in force for a library call.

    An explicit ``root`` narrows the source and is checked against its
    allowlist. The single exception is a root supplied to an *implied* default
    source — no profile declared one for this call — which becomes an ad-hoc
    source confined to that root (see ``adhoc_source``). A source a profile
    actually declared is never escaped this way.
    """
    base = source if source is not None else active_source(profile)
    if root is None:
        return base
    resolved = _absolutize(root, "root")
    if _within(resolved, base.allow):
        return replace(base, roots=(resolved,))
    if source is None and profile is None and not base.declared:
        return adhoc_source(resolved)
    raise SourceError(
        "refusing root %s: outside source %r (allow: %s)"
        % (resolved, base.id, _fmt(base.allow))
    )


def iter_source_files(source: Source) -> List[Tuple[Path, Path]]:
    """(root, file) pairs for every markdown note the source may read.

    Every candidate is resolved and re-checked against the allowlist. A file
    that escapes it fails the whole run rather than being skipped: a silently
    partial corpus you believe is complete is worse than a refusal.
    """
    source.require_enabled()
    pairs: List[Tuple[Path, Path]] = []
    for root in source.roots:
        if not _within(root, source.allow):  # defence in depth
            raise SourceError(
                "source %r: root %s is outside its allowlist (%s)"
                % (source.id, root, _fmt(source.allow))
            )
        if not root.is_dir():
            log.warning("source %r: root %s is not a directory", source.id, root)
            continue
        for path in sorted(root.rglob("*.md")):
            if path.name.startswith(".") or "CLAUDE.md" in path.name:
                continue
            source.check(path)
            pairs.append((root, path))
    return pairs


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


def iter_kb_files(
    root: Optional[Path] = None, source: Optional[Source] = None
) -> List[Path]:
    """Markdown notes of the knowledge base, in stable order and guarded."""
    return [path for _root, path in iter_source_files(resolve_source(source, root))]


def collect_chunks(
    root: Optional[Path] = None,
    config: Optional[ChunkConfig] = None,
    source: Optional[Source] = None,
) -> List[Chunk]:
    """Every chunk of the declared source, deterministically ordered."""
    source = resolve_source(source, root)
    config = resolve_config(config)
    chunks: List[Chunk] = []
    for base, path in iter_source_files(source):
        rel = path.relative_to(base).as_posix()
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


def get_collection(
    create: bool = True,
    name: Optional[str] = None,
    embedding_function=None,
    client=None,
):
    """The collection handle, with the local ONNX embedder attached.

    ``client`` lets a caller that already holds a client reuse it instead of
    building a second one. Two ``PersistentClient`` instances on the same path
    are distinct objects that share one internal ``System``, and therefore one
    collection cache, so state written through one is resolved against cached
    state in the other. Passing the client through keeps a caller's whole
    sequence on a single cache.

    ``embedding_function`` exists for callers that must *time* the embedder.
    Each ONNX embedding function instance caches its own model and ONNX session
    (``cached_property``), so a caller that builds one, loads the model, and then
    lets this function build a second one pays the model load twice —
    and the second load lands inside whatever it was trying to measure. Passing
    the already-warmed instance in makes that cost visible where it was paid.
    Default ``None`` is the existing behaviour, unchanged.
    """
    client = client or get_client()
    name = name or COLLECTION_NAME
    if embedding_function is None:
        embedding_function = _embedding_function()
    if create:
        return client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=name, embedding_function=embedding_function)


def reindex(
    reset: bool = False,
    root: Optional[Path] = None,
    collection_name: Optional[str] = None,
    config: Optional[ChunkConfig] = None,
    source: Optional[Source] = None,
    embedding_function=None,
) -> dict:
    """Rebuild the collection from the declared source. Returns a summary."""
    source = resolve_source(source, root)
    name = collection_name or COLLECTION_NAME
    config = resolve_config(config)
    client = get_client()

    if reset:
        try:
            client.delete_collection(name)
            log.info("dropped collection %s", name)
        except Exception:
            pass  # nothing to drop on a first run

    # Re-created through the *same* client that issued the drop. A second
    # PersistentClient on this path is a distinct object sharing the first one's
    # System-level collection cache, so a get_or_create issued through it is
    # resolved against that shared state and can hand back the dropped
    # collection's UUID; the upsert below then fails with "Collection [uuid]
    # does not exist". Whether the cache returns the dead UUID or a fresh one
    # varies by platform and binding, so the drop and the re-create are kept on
    # one cache-invalidation path rather than left to that race.
    collection = get_collection(
        name=name, embedding_function=embedding_function, client=client
    )

    chunks = collect_chunks(config=config, source=source)
    sources = sorted({c.source for c in chunks})

    if not chunks:
        log.warning("no chunks found under %s", _fmt(source.roots))
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

    collection_name = args.collection
    config: Optional[ChunkConfig] = None
    profile_label = "(defaults)"
    profile: Optional[dict] = None
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

    # The source is resolved strictly here: a human-driven run never falls
    # back to an ad-hoc root. A --root outside the active profile's allowlist,
    # or a source that is declared but not enabled, stops the run.
    try:
        source = active_source(profile)
        if args.root is not None:
            source = source.narrowed_to(args.root)
        source.require_enabled()
    except (ProfileError, SourceError) as exc:
        print("source error: %s" % exc)
        return 2

    if args.stats:
        try:
            chunks = collect_chunks(config=config, source=source)
        except SourceError as exc:
            print("source error: %s" % exc)
            return 2
        by_source: dict = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, 0)
            by_source[chunk.source] += 1
        print("root:   %s" % _fmt(source.roots))
        print("source: %s (%s, publish=%s)" % (source.id, source.origin, source.publish))
        print("allow:  %s" % _fmt(source.allow))
        print("profile: %s" % profile_label)
        print(
            "cutter: max=%d overlap=%d min=%d strategy=%s"
            % (config.max_chars, config.overlap_chars, config.min_chars, config.strategy)
        )
        print("files:  %d" % len(by_source))
        print("chunks: %d" % len(chunks))
        for name, count in sorted(by_source.items()):
            print("  %-60s %3d" % (name, count))
        return 0

    try:
        summary = reindex(
            reset=args.reset,
            collection_name=collection_name,
            config=config,
            source=source,
        )
    except SourceError as exc:
        print("source error: %s" % exc)
        return 2
    print(
        "indexed %d chunks from %d files -> collection '%s' (%d docs)"
        % (summary["chunks"], summary["files"], summary["collection"], summary.get("count", 0))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
