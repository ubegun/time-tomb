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
the body, so re-running the build on unchanged content is a no-op and the git
history alone testifies to when revisions happened.

Usage:
    python manifest.py              # build/refresh manifest.json
    python manifest.py --diff       # compare stored manifest against the body
    python manifest.py --show       # print the stored manifest, resolved
"""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from raglog import (
    KB_ROOT,
    MANIFEST_FILE,
    MANIFEST_MAP_FILE,
    SALT_FILE,
    get_logger,
)

log = get_logger("manifest")

HASH_LEN = 16  # hex chars kept from each sha256
MANIFEST_VERSION = 1
EMPTY = hashlib.sha256(b"rag-skeleton/empty").hexdigest()[:HASH_LEN]


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


def build(root: Optional[Path] = None) -> tuple:
    """Compute (manifest, resolver_map) for the current body."""
    from index_toolbox import collect_chunks

    root = root or KB_ROOT
    salt = load_salt()
    chunks = collect_chunks(root)

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


def diff(root: Optional[Path] = None) -> dict:
    """Compare the stored manifest with a freshly computed one."""
    stored = load_stored()
    current, _ = build(root)
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
    parser.add_argument("--show", action="store_true", help="print stored manifest, resolved")
    parser.add_argument("--root", type=Path, default=None, help="knowledge base root")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.root.resolve() if args.root else KB_ROOT

    if args.show:
        stored = load_stored()
        if stored is None:
            print("no manifest.json")
            return 1
        resolver = load_resolver()
        print("root:   %s" % stored["root"])
        print("files:  %d" % stored["file_count"])
        print("chunks: %d" % stored["chunk_count"])
        for node in stored["tree"]:
            print(
                "  %s  %-50s %3d leaves"
                % (node["hash"], _name_of(node["key"], resolver), node["leaf_count"])
            )
        return 0

    if args.diff:
        report = diff(root)
        if report["status"] == "missing":
            print("no manifest.json yet (current root %s)" % report["current_root"])
            return 1
        if report["status"] == "clean":
            print("no changes (root %s)" % report["current_root"])
            return 0
        print("CHANGED: stored root %s -> current %s" % (report["stored_root"], report["current_root"]))
        for label in ("added", "removed", "changed"):
            for name in report[label]:
                print("  %-8s %s" % (label, name))
        return 1

    manifest, resolver = build(root)
    write(manifest, resolver)
    print(
        "manifest root %s (%d files, %d chunks) -> %s"
        % (manifest["root"], manifest["file_count"], manifest["chunk_count"], MANIFEST_FILE.name)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
