#!/usr/bin/env python
"""Salted Merkle manifest of the knowledge base — the publishable skeleton.

The manifest states how many files and chunks exist and how they are grouped,
and nothing else. Every name and every chunk body is passed through
``sha256(salt || value)`` and truncated, so without the 256-bit salt held in
``.manifest-salt`` an external reader cannot confirm a guessed filename or
chunk: the salt makes a dictionary attack against the short hashes useless.

Local-only artefacts, never published:
    .manifest-salt      the salt itself
    manifest-map.json   key -> real name resolver, for the owner's own tooling

Published artefact:
    manifest.json       root / tree / leaves, a pure function of the content

``manifest.json`` deliberately carries no timestamp: it is a pure function of
the body **under the default chunk geometry**, so re-running the build on
unchanged content is a no-op and the git history alone testifies to when
revisions happened.

Two conditions must both hold before a run may write the published manifest,
and each is checked against the configuration itself, never against a profile
name:

* the active source declares ``publish: true`` — the published root must move
  only when the published body moves (see ``index_toolbox``'s source guard);
* the chunk configuration in force equals the shipped default — a different
  cut of an unchanged body computes a different root, which would move the
  published root while nothing was published.

A run that fails either check is *refused by design* (exit 3), which is a
different thing from failing (exit 2). ``--compute`` prints what such a
configuration would produce without writing anything.

Usage:
    python manifest.py              # build/refresh manifest.json
    python manifest.py --diff       # compare stored manifest against the body
    python manifest.py --compute    # print this configuration's root; writes nothing
    python manifest.py --show       # print the stored manifest, resolved

Exit codes:
    0  ok / clean
    1  dirty or missing — a real answer about the body
    2  error — bad source, bad profile
    3  refused by design — this configuration may not speak for the published
       manifest
"""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from raglog import (
    MANIFEST_FILE,
    MANIFEST_MAP_FILE,
    SALT_FILE,
    get_logger,
)

log = get_logger("manifest")

HASH_LEN = 16  # hex chars kept from each sha256
MANIFEST_VERSION = 1
# Frozen historical seed; renaming moves every manifest hash — must not change.
# It keeps the project's old name ("rag-skeleton") on purpose: this byte string
# feeds every empty-subtree hash, so editing it would move the published
# manifest root while nothing about the body had changed. A rename is not a
# content revision, and the commit history must not claim it was one.
EMPTY = hashlib.sha256(b"rag-skeleton/empty").hexdigest()[:HASH_LEN]

# Exit codes. 3 exists so that "this configuration is not allowed to speak for
# the published manifest" is distinguishable from "something went wrong": a
# caller can act on the first (compute and report) and must stop on the second.
EXIT_OK = 0
EXIT_DIRTY = 1
EXIT_ERROR = 2
EXIT_REFUSED = 3


class GeometryError(RuntimeError):
    """Raised when a non-default chunk geometry tries to write the manifest."""


def _geometry(config) -> str:
    """The geometry as a short label. Only the numbers vary in practice: the
    profile loader refuses any strategy but the implemented one, so a strategy
    mismatch cannot reach here — the comparison still covers it."""
    return "max_chars=%d overlap=%d min=%d" % (
        config.max_chars,
        config.overlap_chars,
        config.min_chars,
    )


def geometry_in_force() -> tuple:
    """``(config, is_default)`` for the chunk configuration this run would use.

    The published manifest describes the body cut the shipped way. Cutting the
    same bytes differently yields a different root, so a run under a different
    geometry must not write ``manifest.json`` — however legitimate its source
    is. The predicate compares the resolved ``ChunkConfig`` against the module
    default; a profile that merely restates the defaults passes, and one that
    changes them is refused. Comparing profile *names* here would be a bug: it
    would bless a name rather than the geometry it happens to carry today.
    """
    from index_toolbox import DEFAULT_CONFIG, resolve_config

    config = resolve_config(None)
    return config, config == DEFAULT_CONFIG


def load_salt(create: bool = True) -> bytes:
    """Read the 256-bit manifest salt, creating it on first use (mode 0600)."""
    if SALT_FILE.exists():
        salt = bytes.fromhex(SALT_FILE.read_text(encoding="utf-8").strip())
        if len(salt) < 32:
            raise ValueError("%s holds fewer than 256 bits" % SALT_FILE)
        return salt

    if not create:
        raise FileNotFoundError("no salt at %s" % SALT_FILE)

    salt = secrets.token_bytes(32)
    # Create with restrictive permissions from the start, never world-readable.
    fd = os.open(str(SALT_FILE), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(salt.hex() + "\n")
    log.warning("generated a new 256-bit salt at %s - back it up, never commit it", SALT_FILE)
    return salt


def salted(salt: bytes, value: str) -> str:
    return hashlib.sha256(salt + value.encode("utf-8")).hexdigest()[:HASH_LEN]


def _pair(left: str, right: str) -> str:
    return hashlib.sha256((left + right).encode("ascii")).hexdigest()[:HASH_LEN]


def merkle_root(hashes: List[str]) -> str:
    """Standard binary Merkle fold; an odd node is paired with itself."""
    if not hashes:
        return EMPTY
    level = list(hashes)
    while len(level) > 1:
        level = [
            _pair(level[i], level[i + 1] if i + 1 < len(level) else level[i])
            for i in range(0, len(level), 2)
        ]
    return level[0]


def build(root: Optional[Path] = None, source=None) -> tuple:
    """Compute (manifest, resolver_map) for the current body."""
    from index_toolbox import collect_chunks

    salt = load_salt()
    chunks = collect_chunks(root=root, source=source)

    by_source: Dict[str, list] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    tree: List[dict] = []
    resolver: Dict[str, object] = {}

    for source in sorted(by_source):
        source_chunks = sorted(by_source[source], key=lambda c: c.index)
        key = salted(salt, source)
        leaves = [salted(salt, chunk.text) for chunk in source_chunks]
        tree.append(
            {
                "key": key,
                "leaf_count": len(leaves),
                "hash": merkle_root(leaves),
                "leaves": leaves,
            }
        )
        resolver[key] = {
            "source": source,
            "leaves": {
                leaf: {"chunk_index": chunk.index, "heading": chunk.heading}
                for leaf, chunk in zip(leaves, source_chunks)
            },
        }

    manifest = {
        "version": MANIFEST_VERSION,
        "algo": "sha256(salt||value)[:%d]" % HASH_LEN,
        "file_count": len(tree),
        "chunk_count": sum(node["leaf_count"] for node in tree),
        "root": merkle_root([node["hash"] for node in tree]),
        "tree": tree,
    }
    return manifest, resolver


def write(manifest: dict, resolver: dict) -> None:
    """Persist the manifest — the one place the published root can move.

    ``main()`` gates on the geometry before it gets here; this repeats the
    check at the mutation point so a library caller cannot move the published
    root by cutting the body a different way. ``build()`` stays permissive on
    purpose: computing a root is harmless, writing one is not.
    """
    from index_toolbox import DEFAULT_CONFIG

    config, is_default = geometry_in_force()
    if not is_default:
        raise GeometryError(
            "refusing to write %s under chunk geometry [%s]: the published "
            "manifest describes the default geometry [%s], and a different cut "
            "of an unchanged body computes a different root. Build it under the "
            "default geometry, or use --compute to see this one's root."
            % (MANIFEST_FILE.name, _geometry(config), _geometry(DEFAULT_CONFIG))
        )

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    MANIFEST_MAP_FILE.write_text(
        json.dumps(resolver, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(MANIFEST_MAP_FILE, 0o600)  # resolver is a local secret
    log.info(
        "manifest root=%s files=%d chunks=%d",
        manifest["root"],
        manifest["file_count"],
        manifest["chunk_count"],
    )


def load_stored() -> Optional[dict]:
    if not MANIFEST_FILE.exists():
        return None
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def load_resolver() -> Dict[str, object]:
    if not MANIFEST_MAP_FILE.exists():
        return {}
    return json.loads(MANIFEST_MAP_FILE.read_text(encoding="utf-8"))


def _name_of(key: str, resolver: Dict[str, object]) -> str:
    entry = resolver.get(key)
    if isinstance(entry, dict) and "source" in entry:
        return str(entry["source"])
    return "<%s>" % key  # unresolvable without the local map — as intended


def diff(root: Optional[Path] = None, source=None) -> dict:
    """Compare the stored manifest with a freshly computed one."""
    stored = load_stored()
    current, _ = build(root, source)
    resolver = load_resolver()

    if stored is None:
        return {
            "status": "missing",
            "message": "no manifest.json yet",
            "current_root": current["root"],
        }

    stored_nodes = {node["key"]: node for node in stored.get("tree", [])}
    current_nodes = {node["key"]: node for node in current["tree"]}

    added = sorted(set(current_nodes) - set(stored_nodes))
    removed = sorted(set(stored_nodes) - set(current_nodes))
    changed = sorted(
        key
        for key in set(stored_nodes) & set(current_nodes)
        if stored_nodes[key]["hash"] != current_nodes[key]["hash"]
    )

    unchanged = not (added or removed or changed)
    return {
        "status": "clean" if unchanged else "dirty",
        "stored_root": stored.get("root"),
        "current_root": current["root"],
        "added": [_name_of(k, resolver) for k in added],
        "removed": [_name_of(k, resolver) for k in removed],
        "changed": [_name_of(k, resolver) for k in changed],
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diff", action="store_true", help="compare stored manifest to body")
    parser.add_argument(
        "--compute",
        action="store_true",
        help="print this configuration's root and counts; writes nothing",
    )
    parser.add_argument("--show", action="store_true", help="print stored manifest, resolved")
    parser.add_argument("--root", type=Path, default=None, help="knowledge base root")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.show:
        stored = load_stored()
        if stored is None:
            print("no manifest.json")
            return EXIT_DIRTY
        resolver = load_resolver()
        print("root:   %s" % stored["root"])
        print("files:  %d" % stored["file_count"])
        print("chunks: %d" % stored["chunk_count"])
        for node in stored["tree"]:
            print(
                "  %s  %-50s %3d leaves"
                % (node["hash"], _name_of(node["key"], resolver), node["leaf_count"])
            )
        return EXIT_OK

    # manifest.json describes the *published* body, cut the shipped way, and
    # nothing else. Two independent things decide whether this run may speak for
    # it: the source (a publish=false source must not move the published root,
    # and a disabled source is refused before anything is read) and the chunk
    # geometry (a different cut computes a different root from the same bytes).
    from index_toolbox import DEFAULT_CONFIG, SourceError, active_source

    try:
        source = active_source()
        if args.root is not None:
            source = source.narrowed_to(args.root)
        source.require_enabled()
    except (RuntimeError, SourceError) as exc:
        print("source error: %s" % exc)
        return EXIT_ERROR

    try:
        config, geometry_is_default = geometry_in_force()
    except RuntimeError as exc:  # a malformed chunking block in the profile
        print("profile error: %s" % exc)
        return EXIT_ERROR

    # --compute answers "what would this configuration produce?" and is exempt
    # from both gates precisely because it cannot mutate anything.
    if args.compute:
        computed, _resolver = build(source=source)
        print(
            "root %s (%d files, %d chunks) - computed, not written"
            % (computed["root"], computed["file_count"], computed["chunk_count"])
        )
        return EXIT_OK

    if not source.publish:
        print(
            "refusing: the active source %r (%s) has publish=false, so it may not "
            "feed %s. The published manifest stays scoped to the default source; "
            "index this source into its own collection instead. Use --compute for "
            "this configuration's root."
            % (source.id, source.origin, MANIFEST_FILE.name)
        )
        return EXIT_REFUSED

    if not geometry_is_default:
        print(
            "refusing: chunk geometry [%s] is not the default [%s]. A different "
            "cut of an unchanged body computes a different root, so this run may "
            "not speak for %s. Use --compute for this configuration's root."
            % (_geometry(config), _geometry(DEFAULT_CONFIG), MANIFEST_FILE.name)
        )
        return EXIT_REFUSED

    if args.diff:
        report = diff(source=source)
        if report["status"] == "missing":
            print("no manifest.json yet (current root %s)" % report["current_root"])
            return EXIT_DIRTY
        if report["status"] == "clean":
            print("no changes (root %s)" % report["current_root"])
            return EXIT_OK
        print("CHANGED: stored root %s -> current %s" % (report["stored_root"], report["current_root"]))
        for label in ("added", "removed", "changed"):
            for name in report[label]:
                print("  %-8s %s" % (label, name))
        return EXIT_DIRTY

    manifest, resolver = build(source=source)
    write(manifest, resolver)
    print(
        "manifest root %s (%d files, %d chunks) -> %s"
        % (manifest["root"], manifest["file_count"], manifest["chunk_count"], MANIFEST_FILE.name)
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
