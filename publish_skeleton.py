#!/usr/bin/env python
"""Rebuild the publishing sandbox: skeleton only, never the body.

The sandbox (``rag/rag-skeleton``) is a real clone of the public repository.
This script makes its working tree exactly match what may be published and
nothing else.

Two properties matter more than convenience here:

* **Allowlist, not denylist.** Only files matching ``PUBLISH_SPEC`` are copied.
  A new file dropped into ``rag/`` later is ignored by default rather than
  published by accident — the failure mode of a denylist is a leak.
* **Content guards, not just names.** Every candidate is scanned for the salt,
  for knowledge-base prose, and for absolute local paths before it is written,
  and the finished sandbox is re-scanned afterwards. A file that passes the
  name check but carries a secret is still refused.

Dry-run is the default; ``--apply`` writes. The script never touches ``.git``
and never commits or pushes — publishing history stays a human decision.

Usage:
    python publish_skeleton.py             # show what would change
    python publish_skeleton.py --apply     # rebuild the sandbox
    python publish_skeleton.py --verify    # audit the sandbox as it stands
"""

import argparse
import filecmp
import hashlib
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from raglog import KB_ROOT, MANIFEST_FILE, RAG_ROOT, SALT_FILE, get_logger

log = get_logger("publish")

SANDBOX = RAG_ROOT / "rag-skeleton"

# What may be published: source path (relative to rag/) -> name in the sandbox.
PUBLISH_SPEC: Dict[str, str] = {
    "manifest.json": "manifest.json",
    "README-public.md": "README.md",
    "index_toolbox.py": "index_toolbox.py",
    "search.py": "search.py",
    "manifest.py": "manifest.py",
    "mcp_server.py": "mcp_server.py",
    "publish_skeleton.py": "publish_skeleton.py",
    "raglog.py": "raglog.py",
}

# Sandbox entries that are managed outside this script and must survive a
# rebuild. Everything else not in PUBLISH_SPEC is stale and gets removed.
PRESERVE = {".git", ".gitignore", "LICENSE"}

# Names that must never appear in the sandbox, whatever else happens.
# Kept generic on purpose: a literal local filename written here would itself
# be published, and a filename is exactly what the manifest exists to hide.
FORBIDDEN_NAMES = {
    ".manifest-salt",
    "manifest-map.json",
    "requirements.txt",
    ".chroma",
    "logs",
    ".venv",
}

# Words that are legitimate vocabulary for a retrieval toolkit even though they
# may also occur in a note's filename. Anything outside this set that appears in
# both a knowledge-base filename and a published file is treated as a leak.
GENERIC_TOKENS = {
    "agent",
    "agents",
    "generated",  # ordinary English ("generated a new salt"), not a topic
    "index",
    "search",
    "manifest",
    "notes",
    "chunk",
    "chunks",
    "case",
    "cases",
    "initial",
    "local",
    "public",
    "readme",
    "skeleton",
    "toolbox",
}

GITIGNORE = """\
# The body never enters this repository — only the skeleton.
.manifest-salt
manifest-map.json
.chroma/
logs/
.venv/
*.tbx
*.log

# Knowledge-base notes: the manifest describes them, the repo never holds them.
KB/
notes/

__pycache__/
*.py[cod]
.DS_Store
"""

_HEX32 = re.compile(r"\b[0-9a-fA-F]{64}\b")  # a 256-bit salt, hex-encoded
_LOCAL_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+")


class LeakError(RuntimeError):
    """Raised when something that must stay local would be published."""


# --- guards ----------------------------------------------------------------


def _body_files() -> List[Path]:
    """The indexed notes — the single definition of "the body" for leak checks.

    Deliberately the same list the indexer uses: files the indexer skips (agent
    instruction files and the like) are not part of the protected body, so
    their names must not become forbidden vocabulary for the tooling.
    """
    from index_toolbox import iter_kb_files

    return iter_kb_files()


def _kb_probes(limit: int = 400) -> List[str]:
    """Distinctive prose fragments from the knowledge base, for leak checks."""
    probes: List[str] = []
    if not KB_ROOT.is_dir():
        return probes
    for path in _body_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            stripped = line.strip()
            # Long prose lines only: short lines and headings collide with
            # ordinary words in the tooling and would produce false alarms.
            if len(stripped) >= 60 and not stripped.startswith(("#", "|", "-", "*", ">")):
                probes.append(stripped[:80])
            if len(probes) >= limit:
                return probes
    return probes


def _kb_name_tokens() -> List[str]:
    """Distinctive words taken from knowledge-base *filenames*.

    Prose probes catch a copied note body; these catch the subtler leak of
    naming a note's topic in a docstring, an example query or a tool
    description. That is a real failure mode: the first version of this
    toolkit shipped a usage example whose sample query was, verbatim, the
    subject of one of the notes it was meant to conceal.
    """
    tokens = set()
    if not KB_ROOT.is_dir():
        return []
    for path in _body_files():
        for token in re.split(r"[^A-Za-z]+", path.stem):
            token = token.lower()
            if len(token) >= 5 and token not in GENERIC_TOKENS:
                tokens.add(token)
    return sorted(tokens)


def scan_content(path: Path, probes: Optional[List[str]] = None) -> List[str]:
    """Return the reasons this file must not be published (empty == clean)."""
    problems: List[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ["unreadable or binary: %s" % path.name]

    if SALT_FILE.exists():
        salt_hex = SALT_FILE.read_text(encoding="utf-8").strip()
        if salt_hex and salt_hex in text:
            problems.append("contains the manifest salt")

    # Any bare 256-bit hex literal is treated as a salt-shaped secret. The
    # manifest's own hashes are 16 chars, so this cannot fire on it.
    for match in _HEX32.findall(text):
        problems.append("contains a 64-char hex literal (%s...)" % match[:8])

    for match in _LOCAL_PATH.findall(text):
        problems.append("contains a local absolute path (%s)" % match)

    for probe in probes if probes is not None else _kb_probes():
        if probe in text:
            problems.append("contains knowledge-base prose (%r...)" % probe[:40])
            break

    lowered = text.lower()
    for token in _kb_name_tokens():
        if re.search(r"\b%s\b" % re.escape(token), lowered):
            problems.append("names a knowledge-base topic (%r)" % token)

    return problems


def plan() -> Tuple[List[tuple], List[Path], List[str]]:
    """Compute (actions, stale entries, problems) without writing anything."""
    problems: List[str] = []
    actions: List[tuple] = []
    probes = _kb_probes()

    if not MANIFEST_FILE.exists():
        problems.append("manifest.json is missing — run manifest.py first")

    for source_name, target_name in sorted(PUBLISH_SPEC.items()):
        source = RAG_ROOT / source_name
        if not source.exists():
            problems.append("missing source: %s" % source_name)
            continue
        for problem in scan_content(source, probes):
            problems.append("%s: %s" % (source_name, problem))

        target = SANDBOX / target_name
        if not target.exists():
            actions.append(("add", source, target))
        elif not filecmp.cmp(str(source), str(target), shallow=False):
            actions.append(("update", source, target))
        else:
            actions.append(("same", source, target))

    stale: List[Path] = []
    if SANDBOX.is_dir():
        published = set(PUBLISH_SPEC.values()) | PRESERVE
        for entry in sorted(SANDBOX.iterdir()):
            if entry.name not in published:
                stale.append(entry)

    return actions, stale, problems


def verify() -> List[str]:
    """Audit the sandbox as it stands on disk. Returns a list of problems."""
    problems: List[str] = []
    if not SANDBOX.is_dir():
        return ["sandbox %s does not exist" % SANDBOX]

    probes = _kb_probes()
    allowed = set(PUBLISH_SPEC.values()) | PRESERVE

    for entry in sorted(SANDBOX.rglob("*")):
        if ".git" in entry.relative_to(SANDBOX).parts:
            continue
        relative = entry.relative_to(SANDBOX)
        if entry.name in FORBIDDEN_NAMES:
            problems.append("FORBIDDEN entry present: %s" % relative)
            continue
        if entry.is_dir():
            problems.append("unexpected directory: %s" % relative)
            continue
        if str(relative) not in allowed:
            problems.append("unexpected file: %s" % relative)
            continue
        if entry.suffix == ".md" and entry.name != "README.md":
            problems.append("stray markdown in the skeleton: %s" % relative)
            continue
        if entry.name == "LICENSE":
            continue  # third-party text, not ours to scan for prose
        for problem in scan_content(entry, probes):
            problems.append("%s: %s" % (relative, problem))

    return problems


# --- apply -----------------------------------------------------------------


def apply(prune: bool = True) -> dict:
    actions, stale, problems = plan()
    if problems:
        raise LeakError("refusing to publish:\n  " + "\n  ".join(problems))

    SANDBOX.mkdir(parents=True, exist_ok=True)

    written = 0
    for verb, source, target in actions:
        if verb == "same":
            continue
        shutil.copyfile(str(source), str(target))
        written += 1
        log.info("%s %s", verb, target.name)

    gitignore = SANDBOX / ".gitignore"
    if not gitignore.exists() or gitignore.read_text(encoding="utf-8") != GITIGNORE:
        gitignore.write_text(GITIGNORE, encoding="utf-8")
        written += 1
        log.info("wrote .gitignore")

    removed = 0
    if prune:
        for entry in stale:
            if entry.name in PRESERVE:
                continue
            if entry.is_dir():
                shutil.rmtree(str(entry))
            else:
                entry.unlink()
            removed += 1
            log.info("removed stale %s", entry.name)

    remaining = verify()
    if remaining:
        raise LeakError("post-publish audit failed:\n  " + "\n  ".join(remaining))

    log.info("sandbox rebuilt: %d written, %d removed", written, removed)
    return {"written": written, "removed": removed, "audited": "clean"}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write changes")
    parser.add_argument("--verify", action="store_true", help="audit the sandbox only")
    parser.add_argument("--no-prune", action="store_true", help="keep unexpected files")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.verify:
        problems = verify()
        if problems:
            print("SANDBOX AUDIT FAILED")
            for problem in problems:
                print("  %s" % problem)
            return 1
        print("sandbox audit clean: %s" % SANDBOX)
        return 0

    if args.apply:
        try:
            summary = apply(prune=not args.no_prune)
        except LeakError as exc:
            print(str(exc))
            return 1
        print(
            "sandbox rebuilt (%d written, %d removed, audit %s): %s"
            % (summary["written"], summary["removed"], summary["audited"], SANDBOX)
        )
        print("review with: git -C %s status" % SANDBOX)
        return 0

    actions, stale, problems = plan()
    print("sandbox: %s" % SANDBOX)
    for verb, source, target in actions:
        print("  %-7s %s" % (verb, target.name))
    for entry in stale:
        print("  %-7s %s" % ("prune", entry.name))
    if problems:
        print("\nWOULD REFUSE:")
        for problem in problems:
            print("  %s" % problem)
        return 1
    print("\ndry run — nothing written. Use --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
