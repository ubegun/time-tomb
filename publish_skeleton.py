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

# What may be published: source path (relative to rag/) -> path in the sandbox.
#
# Single files first. These are named one by one on purpose: the spec is an
# allowlist, and a spec that could be satisfied by a pattern is a spec that can
# be widened by dropping a file into the right directory.
PUBLISH_SPEC: Dict[str, str] = {
    "manifest.json": "manifest.json",
    "README-public.md": "README.md",
    "BENCHMARKS.md": "BENCHMARKS.md",
    "index_toolbox.py": "index_toolbox.py",
    "search.py": "search.py",
    "manifest.py": "manifest.py",
    "mcp_server.py": "mcp_server.py",
    "publish_skeleton.py": "publish_skeleton.py",
    "raglog.py": "raglog.py",
    "bench.py": "bench.py",
    "install.sh": "install.sh",
    "requirements.txt": "requirements.txt",
}

# Directories that may be published *as* directories, each with the glob that
# bounds it. Still an allowlist: only these directories exist, only files
# matching the pattern inside them are candidates, and every candidate goes
# through the same content guards as a file named above.
#
# ``profiles/`` is not published wholesale. Two of the profiles in that
# directory declare sources with absolute local paths, and a profile is a
# configuration file that this repository has no reason to carry unless the
# published bench needs it. The three the bench runs are named, and nothing
# else in the directory is a candidate at all.
PUBLISH_TREES: Tuple[Tuple[str, str], ...] = (
    ("bench/corpus", "*.md"),
    ("bench/corpus", "ground-truth.json"),
    ("bench/results", "*.json"),
)

PUBLISH_PROFILES: Tuple[str, ...] = ("baseline", "chunk-small", "chunk-large")

# Files whose executable bit is part of the artefact.
EXECUTABLE = {"install.sh"}

# Sandbox entries that are managed outside this script and must survive a
# rebuild. Everything else not in the published set is stale and gets removed.
PRESERVE = {".git", ".gitignore", "LICENSE"}

# Names that must never appear in the sandbox, whatever else happens.
# Kept generic on purpose: a literal local filename written here would itself
# be published, and a filename is exactly what the manifest exists to hide.
#
# ``requirements.txt`` used to sit in this set and no longer does. The reason it
# was here was that a requirements file is the sort of thing that carries local
# state — an editable install, a wheel on someone's disk, a private index URL.
# But ``install.sh`` cannot run without it, and the published claim of this
# repository is that a stranger can regenerate the numbers, which needs the
# exact pins rather than a shell script's guess at them. So the blanket ban on
# the *name* is replaced by a contract on the *content*: see
# ``_requirements_problems``. A pin freeze is admitted; anything that reaches
# outside the index is not. That is a narrowing, not a relaxation — the file is
# now checked, where before it was merely absent.
FORBIDDEN_NAMES = {
    ".manifest-salt",
    "manifest-map.json",
    ".chroma",
    "logs",
    ".venv",
}

# Words that are legitimate vocabulary for a retrieval toolkit even though they
# may also occur in a note's filename. Anything outside this set that appears in
# both a knowledge-base filename and a published file is treated as a leak.
#
# The bar for adding one is not "it would be convenient": it is that the word is
# *this repository's own published vocabulary*, so its appearance in a published
# file cannot be evidence that a note's topic leaked. A reader of the public repo
# already sees it in a filename. Every word below meets that bar, and every other
# token taken from a knowledge-base filename stays forbidden.
GENERIC_TOKENS = {
    "agent",
    "agents",
    # ``bench`` is the name of a published directory (``bench/corpus``,
    # ``bench/results``) and of a published file (``bench.py``), and it appears
    # in the one-line reproduce command this repository promises works. It
    # cannot be reworded out of the published set, so it cannot function as a
    # leak signal. A note whose filename contains it is still protected by its
    # other filename tokens and by the prose probes — two independent checks —
    # and this is the only word here that was admitted because a knowledge-base
    # filename started colliding with the toolkit rather than the other way
    # round. That direction is worth recording: the guard fired correctly, and
    # what it found was a name collision, not a leak.
    "bench",
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

# Scratch directories the bench creates and removes.
.bench-corpus-*/
.bench-verify-*/

__pycache__/
*.py[cod]
.DS_Store
"""

_HEX32 = re.compile(r"\b[0-9a-fA-F]{64}\b")  # a 256-bit salt, hex-encoded
_LOCAL_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+")

# A published requirements file may contain comments, blank lines, plain
# ``name==version`` pins and hashes. It may not reach outside the package index:
# no editable installs, no paths, no URLs, no alternate index.
_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9][A-Za-z0-9.*+!_-]*$")
_REQUIREMENTS_REACH_OUT = ("-e ", "--editable", "--index-url", "--extra-index-url",
                           "--find-links", "-f ", "-r ", "--requirement", "file:",
                           "git+", "http://", "https://")


def _requirements_problems(text: str) -> List[str]:
    """The content contract that replaced the ban on the name.

    Every line must be a comment, blank, a hash continuation, or a fully
    pinned requirement. A line that names a location — a path, a URL, an index,
    another requirements file — is refused, because that is the shape a local
    detail takes when it ends up in a pin freeze.
    """
    problems: List[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--hash"):
            continue
        lowered = line.lower()
        for marker in _REQUIREMENTS_REACH_OUT:
            if marker in lowered:
                problems.append(
                    "line %d reaches outside the package index (%r)" % (number, marker.strip())
                )
                break
        else:
            if not _PIN.match(line.split(";")[0].split(" --hash")[0].strip()):
                problems.append("line %d is not a plain name==version pin: %r" % (number, line[:60]))
    return problems


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


def published_map() -> Dict[str, str]:
    """Every publishable source path -> its path in the sandbox.

    One function so that ``plan()``, ``verify()`` and ``apply()`` cannot come to
    different conclusions about what the allowlist says. Directory entries are
    expanded here and nowhere else.
    """
    mapping: Dict[str, str] = dict(PUBLISH_SPEC)
    for name in PUBLISH_PROFILES:
        mapping["profiles/%s.json" % name] = "profiles/%s.json" % name
    for directory, pattern in PUBLISH_TREES:
        base = RAG_ROOT / directory
        if not base.is_dir():
            continue
        for entry in sorted(base.glob(pattern)):
            if not entry.is_file():
                continue
            relative = entry.relative_to(RAG_ROOT).as_posix()
            mapping[relative] = relative
    return mapping


def published_directories(mapping: Dict[str, str]) -> set:
    """Directories the sandbox is allowed to contain, derived from the files."""
    directories = set()
    for target in mapping.values():
        parent = Path(target).parent
        while str(parent) not in (".", ""):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def scan_content(path: Path, probes: Optional[List[str]] = None) -> List[str]:
    """Return the reasons this file must not be published (empty == clean)."""
    problems: List[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ["unreadable or binary: %s" % path.name]

    if path.name == "requirements.txt":
        problems.extend(_requirements_problems(text))

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
    mapping = published_map()

    if not MANIFEST_FILE.exists():
        problems.append("manifest.json is missing — run manifest.py first")

    for directory, pattern in PUBLISH_TREES:
        if not (RAG_ROOT / directory).is_dir():
            problems.append(
                "missing source directory: %s (expected %s)" % (directory, pattern)
            )
        elif not any(key.startswith(directory + "/") for key in mapping):
            problems.append("no %s matched %s/" % (pattern, directory))

    for source_name, target_name in sorted(mapping.items()):
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
        published = set(mapping.values())
        directories = published_directories(mapping)
        for entry in sorted(SANDBOX.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            relative = entry.relative_to(SANDBOX)
            if relative.parts[0] in PRESERVE:
                continue
            key = relative.as_posix()
            if entry.is_dir():
                if key not in directories:
                    stale.append(entry)
            elif key not in published:
                stale.append(entry)

    return actions, stale, problems


# Markdown that may exist in the sandbox, as a rule rather than a list of
# names: the published README, the generated report, and the corpus notes the
# report is computed from. Every other .md is a note that escaped.
def _markdown_allowed(relative: Path) -> bool:
    key = relative.as_posix()
    if key in ("README.md", "BENCHMARKS.md"):
        return True
    return key.startswith("bench/corpus/") and len(relative.parts) == 3


def verify() -> List[str]:
    """Audit the sandbox as it stands on disk. Returns a list of problems."""
    problems: List[str] = []
    if not SANDBOX.is_dir():
        return ["sandbox %s does not exist" % SANDBOX]

    probes = _kb_probes()
    mapping = published_map()
    allowed = set(mapping.values())
    directories = published_directories(mapping)

    for entry in sorted(SANDBOX.rglob("*")):
        relative = entry.relative_to(SANDBOX)
        if relative.parts[0] == ".git":
            continue
        if entry.name in FORBIDDEN_NAMES:
            problems.append("FORBIDDEN entry present: %s" % relative)
            continue
        key = relative.as_posix()
        if entry.is_dir():
            if key not in directories:
                problems.append("unexpected directory: %s" % relative)
            continue
        if key in PRESERVE:
            continue
        if key not in allowed:
            problems.append("unexpected file: %s" % relative)
            continue
        if entry.suffix == ".md" and not _markdown_allowed(relative):
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
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(target))
        # The reproduce line in the README starts with ./install.sh, so the
        # executable bit is part of the artefact, not an accident of the copy.
        if target.name in EXECUTABLE:
            target.chmod(0o755)
        written += 1
        log.info("%s %s", verb, target.relative_to(SANDBOX))

    gitignore = SANDBOX / ".gitignore"
    if not gitignore.exists() or gitignore.read_text(encoding="utf-8") != GITIGNORE:
        gitignore.write_text(GITIGNORE, encoding="utf-8")
        written += 1
        log.info("wrote .gitignore")

    removed = 0
    if prune:
        for entry in stale:
            if entry.relative_to(SANDBOX).parts[0] in PRESERVE:
                continue
            if not entry.exists():
                continue  # already gone with a parent removed earlier
            if entry.is_dir():
                shutil.rmtree(str(entry))
            else:
                entry.unlink()
            removed += 1
            log.info("removed stale %s", entry.relative_to(SANDBOX))

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
        print("  %-7s %s" % (verb, target.relative_to(SANDBOX)))
    for entry in stale:
        print("  %-7s %s" % ("prune", entry.relative_to(SANDBOX)))
    if problems:
        print("\nWOULD REFUSE:")
        for problem in problems:
            print("  %s" % problem)
        return 1
    print("\ndry run — nothing written. Use --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
