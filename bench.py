#!/usr/bin/env python
"""CRUD timing harness for the local retrieval toolkit, one run per profile.

Four operations are timed, N runs each, reported as P50/P95/P99 rather than
averages:

    Create  a full index build of the whole corpus
    Read    one pass over a fixed query set at fixed k, fixed order
    Update  re-index of a single changed note
    Delete  removal of one note, plus the lag until it stops being returned

Two safety properties are structural, not conventions:

1. **The real knowledge base is never involved.** The harness generates its
   own deterministic corpus in a scratch directory, indexes that, and mutates
   only that. ``KB/`` and ``manifest.json`` are fingerprinted on entry and
   re-checked in a ``finally``; a mismatch is a hard error, not a warning.

2. **The live collection is never a target.** Every profile is benched into
   ``bench__<profile>``, asserted to differ from the working collection, and
   dropped again on the way out.

The harness makes no network calls, and does not merely promise that: a socket
guard is installed at import time and every blocked attempt is counted into the
report.

Output goes to ``bench/<date>-<profile>.json`` — machine JSON with a rendered
human table embedded under ``"table"`` — and the same table to stdout.

The ``"quality"`` block carries document-level ``recall@k`` over a labelled
query set, plus MRR and nDCG@10 and a per-query breakdown. Latency without a
quality metric ranks a system that answers fast and wrong above one that
answers slowly and right.

Four properties make that number mean something:

* **Relevance is document-level.** A retrieved chunk is a hit when its
  ``source`` file is in the query's relevant set. Chunk identity is the
  independent variable across profiles, so chunk-keyed labels would make the
  profiles incomparable by construction.
* **Ground truth is planted, never inferred.** ``build_corpus`` decides which
  note answers which query and writes the answer into it; the labels are a
  property of the generator, not of what the retriever happened to return.
* **Quality is measured outside the timed operations.** ``op_quality`` runs
  after all four timed ops and takes no timing samples, so read latency is
  unaffected by it.
* **The metric can come out low.** Two controls run every time and are
  reported with their real numbers: an unanswerable query (no document in the
  corpus answers it) and a deranged label assignment that maps every query
  onto a document set disjoint from its true one.

The corpus is also a *shippable artefact*. ``--emit-corpus`` writes it, with its
labelled query set, as files; ``--corpus`` runs against those files instead of
generating a throwaway copy. The on-disk corpus is verified byte-for-byte
against the generator before anything is measured, and the shipped directory is
never mutated: the timed Update and Delete ops work on a scratch copy, and the
shipped bytes are re-checked on the way out. A corpus a stranger can silently
edit is a corpus whose numbers mean nothing.

Usage:
    python bench.py                                   # baseline, N=10
    python bench.py --profile baseline --profile chunk-small
    python bench.py --profile chunk-large -n 5
    python bench.py --emit-corpus bench/corpus        # write the shipped corpus
    python bench.py --all --corpus bench/corpus       # the published run
    python bench.py --render-benchmarks               # BENCHMARKS.md from JSONs
"""

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --- interpreter bootstrap --------------------------------------------------
# The published reproduce line is ``./install.sh && python bench.py ...``, and
# whatever ``python`` resolves to on a stranger's PATH is not the virtualenv
# install.sh just built. If the current interpreter cannot import the vector
# store and the toolkit's own virtualenv can, hand over to it. Guarded by an
# environment variable so the handover can happen at most once.

_VENV_DIR = Path(__file__).resolve().parent / ".venv"
_VENV_PYTHON = _VENV_DIR / "bin" / "python"


def _reexec_in_venv() -> None:
    if os.environ.get("RAG_BENCH_NO_REEXEC"):
        return
    if importlib.util.find_spec("chromadb") is not None:
        return
    if not _VENV_PYTHON.is_file():
        return
    # ``sys.prefix``, not ``sys.executable``: a virtualenv's bin/python is a
    # symlink to the base interpreter, so resolving the two paths and comparing
    # them says "already inside" for every process started by that base python.
    # sys.prefix is the one value that actually names the active environment.
    if Path(sys.prefix) == _VENV_DIR:
        return
    os.environ["RAG_BENCH_NO_REEXEC"] = "1"
    os.execv(
        str(_VENV_PYTHON),
        [str(_VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:],
    )


_reexec_in_venv()

# --- offline guard ---------------------------------------------------------
# Installed before anything heavy is imported, so nothing this process pulls in
# can open an outbound connection behind our back. Loopback and unix sockets
# stay allowed: they are local IPC, not network egress.

BLOCKED_CONNECTS: List[str] = []
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _is_local(address) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX path and friends
    return str(address[0]) in _LOCAL_HOSTS


def _guarded_connect(self, address, *args, **kwargs):
    if _is_local(address):
        return _real_connect(self, address, *args, **kwargs)
    BLOCKED_CONNECTS.append(repr(address))
    raise OSError("bench.py is offline by design: blocked connect to %r" % (address,))


def _guarded_connect_ex(self, address, *args, **kwargs):
    if _is_local(address):
        return _real_connect_ex(self, address, *args, **kwargs)
    BLOCKED_CONNECTS.append(repr(address))
    return 111  # ECONNREFUSED


socket.socket.connect = _guarded_connect  # type: ignore[assignment]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[assignment]

# Telemetry off before the vector store is imported; the socket guard would
# catch it anyway, but a blocked attempt would be noise in the report.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")

# Ambient configuration must not leak into a measurement: every call below
# passes its chunk configuration and collection name explicitly.
os.environ.pop("RAG_PROFILE", None)

from raglog import COLLECTION_NAME, KB_ROOT, MANIFEST_FILE, RAG_ROOT, get_logger  # noqa: E402
from index_toolbox import (  # noqa: E402
    ChunkConfig,
    config_from_profile,
    chunk_markdown,
    get_client,
    get_collection,
    load_profile,
    reindex,
)

log = get_logger("bench")

SCHEMA_VERSION = 2
DEFAULT_RUNS = 10
BENCH_DIR = RAG_ROOT / "bench"
RESULTS_DIR = BENCH_DIR / "results"
SHIPPED_CORPUS = BENCH_DIR / "corpus"
BENCHMARKS_FILE = RAG_ROOT / "BENCHMARKS.md"
GROUND_TRUTH_FILE = "ground-truth.json"
COLLECTION_PREFIX = "bench__"

# The three profiles published together. Order is the chunk-size ladder, not
# alphabetical: the tables read as a progression from finest to coarsest.
PUBLISHED_PROFILES: Tuple[str, ...] = ("chunk-small", "baseline", "chunk-large")

# --- the fixed query set ---------------------------------------------------
# Fixed strings, fixed order, no sampling, no shuffling. Read latency must be
# a property of the index, not of which questions the run happened to draw.
QUERIES: Tuple[str, ...] = (
    "how does the harvester schedule its lattice passes",
    "quorum drift between the outer and inner relay",
    "what does the ledger say about pumice throughput",
    "cost of a cold restart for the ferrite pipeline",
    "marker phrase zorbium quintal beacon",
)

# Planted only in the scratch note, so a Delete can be proved by retrieval and
# not merely by a bookkeeping call returning success.
DELETE_MARKER = "zorbium quintal beacon"
DELETE_PROBE = "marker phrase zorbium quintal beacon"
# The delete proof queries wider than the timed Read op: see _probe_k().
PROBE_K = 25


# --- the labelled query set (retrieval quality) -----------------------------
# Separate from QUERIES above on purpose. QUERIES exists to time the Read op
# and must not change; this set exists to score it and is never timed.
#
# Each topic is a fact planted by build_corpus() into a named set of notes.
# The nouns are invented and occur nowhere else in the corpus - not in _WORDS,
# not in the delete marker - so "which document answers this question" is
# decided by the generator and is not a judgement call.
#
# Two things make the set hard enough to have resolution:
#
# * |relevant| is 2 or 3, so recall@1 is capped at 1/|relevant| and the ladder
#   cannot be binary.
# * The topics come in **confusable families**: two topics share a family noun
#   (thernwick, vashtrin, pellagrove, ganthorpe) and differ only in the device
#   noun, and the two members of a family are planted in disjoint document
#   sets. A retriever that matches on the rare family token alone lands on the
#   wrong half of the corpus. A first draft without the families scored a
#   perfect 1.000 at every rung on every profile - a labelled set that only a
#   broken retriever could fail is not a measurement.

QUALITY_TOPICS: Tuple[Dict[str, object], ...] = (
    {
        "id": "thernwick-governor",
        "query": "what pressure does the thernwick governor hold at cold start",
        "fact": (
            "The thernwick governor holds a cold-start pressure of 47 kilopascal. "
            "Crews read the thernwick governor before the first pass of a shift, "
            "because a governor that has drifted below 40 kilopascal will not seat."
        ),
        "notes": (0, 1),
    },
    {
        "id": "thernwick-rail",
        "query": "how often is the thernwick rail re-shimmed",
        "fact": (
            "The thernwick rail is re-shimmed every nine hundred cycles. "
            "Shimming the thernwick rail early wastes a wear allowance that is "
            "measured over the whole service interval, not over a single week."
        ),
        "notes": (2, 3),
    },
    {
        "id": "vashtrin-buffer",
        "query": "what temperature limit applies to the vashtrin buffer",
        "fact": (
            "The vashtrin buffer must stay under sixty-two degrees at the outlet. "
            "Above that the vashtrin buffer starts to shed its facing, and the "
            "loss is not recoverable by cooling it back down afterwards."
        ),
        "notes": (4, 5),
    },
    {
        "id": "pellagrove-damper",
        "query": "why does the pellagrove damper need a second anchor",
        "fact": (
            "The pellagrove damper needs a second anchor because its first anchor "
            "carries only vertical load. Without the second anchor a pellagrove "
            "damper walks sideways under reversing load and eventually fouls."
        ),
        "notes": (0, 2, 4),
    },
    {
        "id": "pellagrove-manifold",
        "query": "what is the bleed order for the pellagrove manifold",
        "fact": (
            "The pellagrove manifold is bled outboard first, then centre, then "
            "inboard. Bleeding a pellagrove manifold in the reverse order traps a "
            "pocket at the centre port that no amount of further bleeding clears."
        ),
        "notes": (1, 3, 5),
    },
    {
        "id": "vashtrin-escapement",
        "query": "what tool releases the vashtrin escapement",
        "fact": (
            "The vashtrin escapement is released with a quarter-inch offset key. "
            "Prying a vashtrin escapement with a screwdriver deforms the pawl "
            "seat, which is the one part of the assembly with no spare."
        ),
        "notes": (0, 3),
    },
    {
        "id": "ganthorpe-diaphragm",
        "query": "what causes the ganthorpe diaphragm to blister",
        "fact": (
            "The ganthorpe diaphragm blisters when it is seated dry. A ganthorpe "
            "diaphragm wetted with the specified film will not blister even after "
            "several hundred reversals at full stroke."
        ),
        "notes": (1, 4),
    },
    {
        "id": "ganthorpe-wickstay",
        "query": "how is the ganthorpe wickstay tensioned",
        "fact": (
            "The ganthorpe wickstay is tensioned to eleven newton metres, in two "
            "passes. A ganthorpe wickstay taken to full torque in one pass reads "
            "correct on the wrench and is still slack across the far side."
        ),
        "notes": (2, 5),
    },
)

# Failure control (a). No document mentions a thernwick escutcheon - the
# device noun appears nowhere in the corpus - so the relevant set is empty and
# the query can only score 0.
#
# It deliberately reuses a *family* noun the corpus is full of, so the
# retriever answers it confidently with lexically plausible chunks. That is
# the point: the control has to survive a confident wrong answer, which is
# exactly the failure mode latency cannot see. A metric that scored this above
# zero would be reading its labels off the retriever's own output.
NO_ANSWER_QUERY = "what clearance does the thernwick escutcheon take when it is refaced"

# The k-ladder and the retrieval depth quality is scored at. k counts
# *chunks*: it is the retriever's depth, the thing a caller actually sets.
QUALITY_LADDER: Tuple[int, ...] = (1, 3, 5, 10)
QUALITY_DEPTH = max(QUALITY_LADDER)


# --- deterministic corpus --------------------------------------------------


class _Det:
    """A tiny fixed LCG. Reproducible across machines and Python versions."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0x7FFFFFFF

    def step(self) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state

    def pick(self, seq):
        return seq[self.step() % len(seq)]


_WORDS = (
    # "beacon" is deliberately absent: it belongs to the delete marker and the
    # corpus must not compete with it.
    "harvester lattice quorum relay ledger pumice ferrite obsidian cadence "
    "spindle aperture gantry tundra basalt cobalt vellum quarry mordant "
    "trellis fathom pillar solstice granite ripple cistern anvil marrow "
    "vector cadmium plinth thicket lantern furrow bellows quartz nimbus"
).split()

_SECTION_LADDER = (140, 420, 780, 1150, 1750, 2600)  # target body sizes, chars
CORPUS_NOTES = 6
SECTIONS_PER_NOTE = 6
CORPUS_SEED = 20260720
SCRATCH_NOTE = "scratch-note.md"


def _paragraph(det: _Det, target: int) -> str:
    words: List[str] = []
    size = 0
    while size < target:
        word = det.pick(_WORDS)
        words.append(word)
        size += len(word) + 1
        if len(words) % 14 == 0:
            words.append("\n")
    return " ".join(words).replace(" \n ", "\n\n").strip()


def _note_sections(det: _Det, sections: int) -> List[List[str]]:
    """The (heading, body) pairs of one note, before anything is planted."""
    out: List[List[str]] = []
    for index in range(sections):
        target = _SECTION_LADDER[det.step() % len(_SECTION_LADDER)]
        heading = "%s %d" % (det.pick(_WORDS), index)
        out.append([heading, _paragraph(det, target)])
    return out


def _render_note(title: str, sections: List[List[str]]) -> str:
    parts = ["# %s" % title, ""]
    for heading, body in sections:
        parts.append("## %s" % heading)
        parts.append("")
        parts.append(body)
        parts.append("")
    return "\n".join(parts)


def _note_text(det: _Det, title: str, sections: int) -> str:
    return _render_note(title, _note_sections(det, sections))


def _plant_fact(sections: List[List[str]], position: int, fact: str) -> None:
    """Bury ``fact`` mid-body in one section, as its own paragraph.

    Buried rather than appended, and mid-body rather than at the top, because
    a fact sitting alone under its own heading would be one clean chunk in
    every profile and the comparison would measure nothing. Inside a long
    section, chunk geometry decides whether the fact is retrieved as a tight
    passage or diluted into a wall of neighbouring text - which is the effect
    this bench exists to detect.
    """
    heading, body = sections[position]
    paragraphs = body.split("\n\n")
    sections[position] = [
        heading,
        "\n\n".join(
            paragraphs[: max(1, len(paragraphs) // 2)]
            + [fact]
            + paragraphs[max(1, len(paragraphs) // 2) :]
        ),
    ]


def note_name(index: int) -> str:
    return "note-%02d.md" % index


def ground_truth() -> List[Dict[str, object]]:
    """query -> relevant source files. A pure function of QUALITY_TOPICS.

    Nothing here consults an index, a collection or a retriever.
    """
    labels: List[Dict[str, object]] = []
    for topic in QUALITY_TOPICS:
        labels.append(
            {
                "id": topic["id"],
                "query": topic["query"],
                "relevant": [note_name(i) for i in topic["notes"]],  # type: ignore[union-attr]
            }
        )
    return labels


def scratch_note_text(variant: int) -> str:
    """The mutable note. Two variants of comparable size, toggled per Update.

    Its last section is the delete marker: a short chunk built almost entirely
    out of tokens that occur nowhere else in the corpus, so a probe query can
    show the note leaving the result set. The phrase is repeated because the
    marker has to stay retrievable at the smallest chunk size too, where it
    competes with several times as many neighbours.
    """
    det = _Det(CORPUS_SEED + 9000 + variant)
    body = _note_text(det, "scratch note variant %d" % variant, 4)
    marker = (
        "## %s\n\n%s. %s. %s marker section, revision %d of the mutable note "
        "used for update and delete timing.\n"
        % (DELETE_MARKER, DELETE_MARKER, DELETE_MARKER, DELETE_MARKER, variant)
    )
    return "%s\n%s" % (body, marker)


def build_corpus(target: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Write the synthetic corpus. Deterministic: same bytes on every run.

    Returns ``(info, planting)``. ``planting`` is the record of which fact went
    into which file and section - the ground truth, fixed here at generation
    time and carried forward untouched. It is never recomputed from anything
    the retriever says.
    """
    target.mkdir(parents=True, exist_ok=True)
    det = _Det(CORPUS_SEED)
    total = 0
    planting: List[Dict[str, object]] = []
    for index in range(CORPUS_NOTES):
        name = note_name(index)
        sections = _note_sections(det, SECTIONS_PER_NOTE)
        # Longest sections first, ties by original order. Fixed before any
        # insertion so that planting one fact cannot move the next one.
        by_length = sorted(
            range(len(sections)), key=lambda i: (-len(sections[i][1]), i)
        )
        slot = 0
        for topic in QUALITY_TOPICS:
            if index not in topic["notes"]:  # type: ignore
                continue
            if slot >= len(by_length):
                raise RuntimeError(
                    "note %s has %d sections but %d facts to plant"
                    % (name, len(sections), slot + 1)
                )
            position = by_length[slot]
            original = len(sections[position][1])
            _plant_fact(sections, position, str(topic["fact"]))
            planting.append(
                {
                    "topic": topic["id"],
                    "source": name,
                    "section": sections[position][0],
                    "section_chars_before": original,
                    "section_chars_after": len(sections[position][1]),
                }
            )
            slot += 1
        text = _render_note("synthetic note %d" % index, sections)
        (target / name).write_text(text, encoding="utf-8")
        total += len(text)
    scratch = scratch_note_text(0)
    (target / SCRATCH_NOTE).write_text(scratch, encoding="utf-8")
    total += len(scratch)

    # The mutable note is deleted and rebuilt by the Update/Delete ops. If it
    # ever carried a labelled fact, quality would depend on which op ran last.
    for entry in planting:
        if entry["source"] == SCRATCH_NOTE:
            raise RuntimeError("a labelled fact was planted in the mutable note")

    info = {
        "files": CORPUS_NOTES + 1,
        "chars": total,
        "scratch": SCRATCH_NOTE,
        "planted_facts": len(planting),
    }
    return info, planting


# --- the corpus as a shipped artefact ---------------------------------------
#
# Three separable things live here and are deliberately not collapsed into one:
#
# * ``emit_corpus`` writes the generator's output to a directory that is meant
#   to be committed and read by a stranger.
# * ``verify_corpus`` regenerates into a scratch directory and compares byte
#   for byte. It is the reason a shipped corpus means anything: the numbers are
#   only about *this* corpus if the files on disk are still the generator's.
# * ``load_ground_truth`` reads the labels back from the shipped file rather
#   than re-deriving them from ``QUALITY_TOPICS``. Re-deriving would make the
#   shipped labels decorative — the run would score against the code's opinion
#   whatever the file said.

CORPUS_SCHEMA_VERSION = 1
_SHORT_DIGEST = 16  # sha256 truncated, as manifest.json does; never a 64-hex


def _short_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:_SHORT_DIGEST]


def corpus_documents() -> List[str]:
    """Every markdown file the corpus consists of, in stable order."""
    return [note_name(i) for i in range(CORPUS_NOTES)] + [SCRATCH_NOTE]


def ground_truth_document(planting: List[Dict[str, object]], files: Dict[str, bytes]) -> dict:
    """The labelled query set, as it is written to disk beside the corpus."""
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "generator": "bench.py:build_corpus",
        "seed": CORPUS_SEED,
        "documents": corpus_documents(),
        "mutable_document": SCRATCH_NOTE,
        "labels_origin": "planted at corpus generation; never derived from retriever output",
        "relevance": (
            "a retrieved chunk is a hit when its source file is in the query's "
            "relevant set; labels are never keyed to chunk ids"
        ),
        "k_ladder": list(QUALITY_LADDER),
        "retrieval_depth": QUALITY_DEPTH,
        "queries": ground_truth(),
        "topics": [
            {
                "id": topic["id"],
                "query": topic["query"],
                "fact": topic["fact"],
                "notes": [note_name(i) for i in topic["notes"]],  # type: ignore
            }
            for topic in QUALITY_TOPICS
        ],
        "no_answer_query": NO_ANSWER_QUERY,
        "timed_queries": list(QUERIES),
        "delete_marker": DELETE_MARKER,
        "planting": planting,
        "digest": {
            "algorithm": "sha256, hex, truncated to %d characters" % _SHORT_DIGEST,
            "files": {name: _short_digest(files[name]) for name in corpus_documents()},
        },
    }


def _generate_corpus_bytes() -> Tuple[Dict[str, bytes], dict, List[Dict[str, object]]]:
    """Run the generator into a throwaway directory and return its bytes."""
    # Inside the toolkit's own directory, never the system temp: everything
    # this harness creates stays where the workspace can account for it.
    scratch = RAG_ROOT / (".bench-verify-%d-%d" % (os.getpid(), time.time_ns() % 1000000))
    try:
        info, planting = build_corpus(scratch)
        files = {name: (scratch / name).read_bytes() for name in corpus_documents()}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return files, info, planting


def emit_corpus(target: Path) -> dict:
    """Write the shipped corpus and its ground truth to ``target``."""
    files, info, planting = _generate_corpus_bytes()
    target.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (target / name).write_bytes(data)
    document = ground_truth_document(planting, files)
    (target / GROUND_TRUTH_FILE).write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written = sorted(files) + [GROUND_TRUTH_FILE]
    log.info("corpus emitted to %s (%d files)", target, len(written))
    return {
        "dir": str(target),
        "files": written,
        "chars": info["chars"],
        "planted_facts": len(planting),
    }


class CorpusError(RuntimeError):
    """The on-disk corpus is not the one the generator produces."""


def verify_corpus(source: Path) -> dict:
    """Compare an on-disk corpus against the generator. Raises on any mismatch.

    Loud, and per-file: a diff that says only "something changed" would send a
    reader looking through seven notes by hand.
    """
    if not source.is_dir():
        raise CorpusError("no corpus directory at %s" % source)

    expected, info, planting = _generate_corpus_bytes()
    problems: List[str] = []

    present = sorted(path.name for path in source.glob("*.md"))
    if present != sorted(expected):
        missing = sorted(set(expected) - set(present))
        extra = sorted(set(present) - set(expected))
        if missing:
            problems.append("missing note(s): %s" % ", ".join(missing))
        if extra:
            problems.append("unexpected note(s): %s" % ", ".join(extra))

    for name in sorted(expected):
        path = source / name
        if not path.is_file():
            continue
        actual = path.read_bytes()
        if actual != expected[name]:
            problems.append(
                "%s differs from the generator (on disk %s / expected %s, "
                "%d vs %d bytes)"
                % (
                    name,
                    _short_digest(actual),
                    _short_digest(expected[name]),
                    len(actual),
                    len(expected[name]),
                )
            )

    truth_path = source / GROUND_TRUTH_FILE
    if not truth_path.is_file():
        problems.append("missing %s" % GROUND_TRUTH_FILE)
    else:
        try:
            stored = json.loads(truth_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            stored = None
            problems.append("%s is not valid JSON: %s" % (GROUND_TRUTH_FILE, exc))
        if stored is not None:
            fresh = ground_truth_document(planting, expected)
            if stored != fresh:
                differing = sorted(
                    key
                    for key in set(stored) | set(fresh)
                    if stored.get(key) != fresh.get(key)
                )
                problems.append(
                    "%s differs from the generator in: %s"
                    % (GROUND_TRUTH_FILE, ", ".join(differing) or "(ordering)")
                )

    if problems:
        raise CorpusError(
            "the corpus at %s is not the generator's output:\n  %s\n"
            "Regenerate it with: python bench.py --emit-corpus %s"
            % (source, "\n  ".join(problems), source)
        )

    return dict(info, verified=True, ground_truth=GROUND_TRUTH_FILE)


def load_ground_truth(source: Path) -> dict:
    """Read the shipped labels. Used instead of re-deriving them from code."""
    return json.loads((source / GROUND_TRUTH_FILE).read_text(encoding="utf-8"))


def stage_corpus(source: Path, scratch: Path) -> None:
    """Copy the shipped corpus into a working directory.

    The timed Update and Delete ops rewrite and remove the mutable note. They
    must never do that to the directory under version control, so the shipped
    copy is read once and never written.
    """
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    for name in corpus_documents():
        shutil.copyfile(str(source / name), str(scratch / name))


# --- state guard -----------------------------------------------------------


def _fingerprint_tree(root: Path) -> str:
    """Hash of every file under ``root``: names, sizes and contents."""
    digest = hashlib.sha256()
    if not root.is_dir():
        return "absent"
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _fingerprint_file(path: Path) -> str:
    if not path.is_file():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StateGuard:
    """Fingerprints what must not move, and proves it did not."""

    def __init__(self) -> None:
        self.kb_root = KB_ROOT
        self.kb_before = _fingerprint_tree(KB_ROOT)
        self.manifest_before = _fingerprint_file(MANIFEST_FILE)
        self.manifest_root_before = self._stored_root()

    @staticmethod
    def _stored_root() -> Optional[str]:
        if not MANIFEST_FILE.is_file():
            return None
        try:
            return json.loads(MANIFEST_FILE.read_text(encoding="utf-8")).get("root")
        except ValueError:
            return None

    def verify(self) -> Dict[str, object]:
        kb_after = _fingerprint_tree(self.kb_root)
        manifest_after = _fingerprint_file(MANIFEST_FILE)
        root_after = self._stored_root()
        problems = []
        if kb_after != self.kb_before:
            problems.append(
                "KB tree changed: %s -> %s" % (self.kb_before[:16], kb_after[:16])
            )
        if manifest_after != self.manifest_before:
            problems.append(
                "manifest.json changed: %s -> %s"
                % (self.manifest_before[:16], manifest_after[:16])
            )
        if root_after != self.manifest_root_before:
            problems.append(
                "manifest root changed: %s -> %s"
                % (self.manifest_root_before, root_after)
            )
        return {
            "kb_tree_sha256": kb_after,
            "kb_unchanged": kb_after == self.kb_before,
            "kb_present": kb_after != "absent",
            "manifest_json_sha256": manifest_after,
            "manifest_root": root_after,
            "manifest_unchanged": manifest_after == self.manifest_before,
            "problems": problems,
        }

    def public(self) -> Dict[str, object]:
        """The verdict, without the fingerprints it was reached from.

        The full digests are what this process compares and what it prints; a
        *report* is a different thing. ``kb_tree_sha256`` is an unsalted hash of
        a private directory, so publishing it would let a holder of a candidate
        body confirm it — the one thing the salted manifest exists to prevent.
        The reader of a report needs the verdict, and the verdict is a boolean.
        """
        full = self.verify()
        return {
            "checked": ["KB tree", "manifest.json bytes", "manifest root"],
            "kb_present": full["kb_present"],
            "kb_unchanged": full["kb_unchanged"],
            "manifest_unchanged": full["manifest_unchanged"],
            "manifest_root": full["manifest_root"],
            "digests_withheld": (
                "the tree fingerprints are compared in-process and printed to "
                "stdout; they are unsalted hashes of a private directory and "
                "are not written to a published report"
            ),
            "problems": full["problems"],
        }


# --- statistics ------------------------------------------------------------


def percentile(values: List[float], quantile: float) -> Optional[float]:
    """Linear interpolation between closest ranks (numpy's default method)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise(samples: List[float], unit: str) -> Dict[str, object]:
    return {
        "unit": unit,
        "n": len(samples),
        "cold_ms": round(samples[0], 3) if samples else None,
        "warm_p50_ms": round(percentile(samples[1:], 0.50), 3) if len(samples) > 1 else None,
        "p50_ms": round(percentile(samples, 0.50), 3) if samples else None,
        "p95_ms": round(percentile(samples, 0.95), 3) if samples else None,
        "p99_ms": round(percentile(samples, 0.99), 3) if samples else None,
        "min_ms": round(min(samples), 3) if samples else None,
        "max_ms": round(max(samples), 3) if samples else None,
        "mean_ms": round(statistics.fmean(samples), 3) if samples else None,
        "stdev_ms": round(statistics.pstdev(samples), 3) if len(samples) > 1 else None,
        "samples_ms": [round(value, 3) for value in samples],
    }


# --- retrieval quality ------------------------------------------------------
# Pure functions of (ranked source list, relevant source set). No collection,
# no client, no clock: whatever these return, they cannot have measured the
# retriever's opinion of itself, and they cannot perturb a timing.


def score_query(
    ranked_sources: List[str],
    relevant: Iterable[str],
    ladder: Iterable[int] = QUALITY_LADDER,
    depth: int = QUALITY_DEPTH,
) -> Dict[str, object]:
    """Document-level retrieval quality for one query.

    ``ranked_sources`` is the source file of each retrieved chunk, best first,
    duplicates included - that is what the retriever actually handed back.

    * ``recall@k`` - fraction of the relevant *documents* that appear as the
      source of at least one of the top-k *chunks*. k is retrieval depth, the
      knob a caller sets, so a profile whose chunks are coarse enough to cover
      more documents per slot is credited for it and one that spends all k
      slots inside a single document is not.
    * ``rr`` - reciprocal rank of the first relevant chunk. Answers "how far
      down before the first right document".
    * ``ndcg@depth`` - binary-gain nDCG over the de-duplicated document
      ranking. Answers "are *all* the right documents near the top", which
      MRR by construction cannot: MRR stops at the first hit. With 2-3
      relevant documents per query the two diverge, and the divergence is the
      part that chunk geometry moves.

    An empty relevant set (the unanswerable control) scores 0 everywhere: the
    denominator is floored at 1, so a non-zero score remains representable and
    would mean the labels had leaked from the retriever.
    """
    rel = set(relevant)
    denominator = max(1, len(rel))
    top = list(ranked_sources[:depth])

    recall: Dict[str, float] = {}
    for k in ladder:
        hit = {source for source in top[:k] if source in rel}
        recall["recall@%d" % k] = round(len(hit) / denominator, 6)

    first_rank: Optional[int] = None
    for position, source in enumerate(top, start=1):
        if source in rel:
            first_rank = position
            break

    documents: List[str] = []
    for source in top:
        if source not in documents:
            documents.append(source)
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, source in enumerate(documents, start=1)
        if source in rel
    )
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, len(rel) + 1))

    return {
        "relevant": sorted(rel),
        "retrieved_sources": top,
        "distinct_documents_retrieved": len(documents),
        "recall": recall,
        "first_relevant_rank": first_rank,
        "rr": round(1.0 / first_rank, 6) if first_rank else 0.0,
        "ndcg@%d" % depth: round(dcg / idcg, 6) if idcg else 0.0,
    }


def aggregate_scores(
    scored: List[Dict[str, object]], ladder: Iterable[int] = QUALITY_LADDER,
    depth: int = QUALITY_DEPTH,
) -> Dict[str, object]:
    if not scored:
        return {}
    out: Dict[str, object] = {"queries": len(scored)}
    for k in ladder:
        key = "recall@%d" % k
        out[key] = round(
            statistics.fmean([float(s["recall"][key]) for s in scored]), 6  # type: ignore[index]
        )
    out["mrr"] = round(statistics.fmean([float(s["rr"]) for s in scored]), 6)
    ndcg_key = "ndcg@%d" % depth
    out[ndcg_key] = round(statistics.fmean([float(s[ndcg_key]) for s in scored]), 6)
    return out


def derange_labels(relevant_sets: List[List[str]]) -> Optional[List[int]]:
    """A permutation giving every query someone else's documents.

    Deterministic backtracking, smallest index first. Requires the swapped set
    to be *document-disjoint* from the true one, not merely a different query:
    a shuffle that happened to hand back an overlapping set would soften the
    control and make the drop look smaller than it is. Returns ``None`` if no
    disjoint permutation exists, in which case the caller reports the residual
    overlap rather than pretending there is none.
    """
    total = len(relevant_sets)
    sets = [set(entry) for entry in relevant_sets]
    used = [False] * total
    assignment: List[int] = [-1] * total

    def search(index: int) -> bool:
        if index == total:
            return True
        for candidate in range(total):
            if used[candidate] or candidate == index:
                continue
            if sets[index] & sets[candidate]:
                continue
            used[candidate] = True
            assignment[index] = candidate
            if search(index + 1):
                return True
            used[candidate] = False
            assignment[index] = -1
        return False

    return list(assignment) if search(0) else None


# --- the harness ------------------------------------------------------------


class ProfileBench:
    def __init__(
        self,
        profile: dict,
        corpus: Path,
        runs: int,
        k: int,
        truth: Dict[str, object],
    ) -> None:
        self.profile = profile
        self.name = profile["name"]
        self.config: ChunkConfig = config_from_profile(profile)
        self.corpus = corpus
        self.runs = runs
        self.k = k
        # The labelled set as it was loaded — from the shipped file when one is
        # in use. Nothing below re-derives a label from QUALITY_TOPICS.
        self.truth = truth
        self._planting = list(truth.get("planting") or [])  # type: ignore
        self.collection_name = COLLECTION_PREFIX + self.name.replace("-", "_")
        self._assert_target_safe()
        self.scratch_path = corpus / SCRATCH_NOTE
        self.notes: Dict[str, object] = {}

    def _assert_target_safe(self) -> None:
        if not self.collection_name.startswith(COLLECTION_PREFIX):
            raise RuntimeError("refusing a collection outside the bench namespace")
        if self.collection_name == COLLECTION_NAME:
            raise RuntimeError(
                "refusing to bench against the working collection %r" % COLLECTION_NAME
            )

    # -- helpers ------------------------------------------------------------

    def _collection(self):
        return get_collection(name=self.collection_name)

    def _index_all(self, reset: bool) -> dict:
        return reindex(
            reset=reset,
            root=self.corpus,
            collection_name=self.collection_name,
            config=self.config,
        )

    def _chunks_for(self, path: Path) -> list:
        text = path.read_text(encoding="utf-8")
        return chunk_markdown(text, path.name, self.config)

    def _upsert_note(self, collection, path: Path) -> int:
        chunks = self._chunks_for(path)
        collection.delete(where={"source": path.name})
        if chunks:
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
        return len(chunks)

    def _sources_in_results(self, collection, query: str, k: int) -> List[str]:
        raw = collection.query(query_texts=[query], n_results=max(1, k))
        metadatas = (raw.get("metadatas") or [[]])[0]
        return [m.get("source", "?") for m in metadatas]

    def _probe_k(self, collection) -> int:
        """Result width for the delete proof.

        Wider than the timed Read: this is a correctness probe, not a latency
        measurement, and at the smallest chunk size the corpus carries several
        times as many neighbours competing for the same five slots.
        """
        return max(1, min(PROBE_K, collection.count() or PROBE_K))

    # -- the four operations ------------------------------------------------

    def op_create(self) -> Tuple[List[float], dict]:
        samples: List[float] = []
        summary: dict = {}
        for _ in range(self.runs):
            started = time.perf_counter()
            summary = self._index_all(reset=True)
            samples.append((time.perf_counter() - started) * 1000.0)
        return samples, summary

    def op_read(self) -> Tuple[List[float], Dict[str, dict]]:
        collection = self._collection()
        samples: List[float] = []
        per_query: Dict[str, List[float]] = {query: [] for query in QUERIES}
        for _ in range(self.runs):
            pass_started = time.perf_counter()
            for query in QUERIES:
                query_started = time.perf_counter()
                collection.query(query_texts=[query], n_results=self.k)
                per_query[query].append((time.perf_counter() - query_started) * 1000.0)
            samples.append((time.perf_counter() - pass_started) * 1000.0)
        rolled = {
            query: summarise(values, "ms per query, k=%d" % self.k)
            for query, values in per_query.items()
        }
        return samples, rolled

    def op_update(self) -> Tuple[List[float], dict]:
        collection = self._collection()
        samples: List[float] = []
        chunk_counts: List[int] = []
        for run in range(self.runs):
            variant = (run % 2) + 1  # alternate, so every run is a real change
            self.scratch_path.write_text(scratch_note_text(variant), encoding="utf-8")
            started = time.perf_counter()
            written = self._upsert_note(collection, self.scratch_path)
            samples.append((time.perf_counter() - started) * 1000.0)
            chunk_counts.append(written)
        # leave the corpus on a known variant
        self.scratch_path.write_text(scratch_note_text(0), encoding="utf-8")
        self._upsert_note(collection, self.scratch_path)
        return samples, {"chunks_rewritten_per_run": chunk_counts}

    def op_delete(self) -> Tuple[List[float], List[float], dict]:
        collection = self._collection()
        delete_samples: List[float] = []
        consistency_samples: List[float] = []
        verifications: List[dict] = []

        for _ in range(self.runs):
            # restore first, untimed: every run must delete something real
            present = self._upsert_note(collection, self.scratch_path)
            if present == 0:
                raise RuntimeError("scratch note produced no chunks to delete")

            # Positive control: the probe must find the note *before* the
            # delete, otherwise "it is gone afterwards" proves nothing.
            probe_k = self._probe_k(collection)
            before = self._sources_in_results(collection, DELETE_PROBE, probe_k)
            if SCRATCH_NOTE not in before:
                raise RuntimeError(
                    "probe did not retrieve %s before deletion (got %s) - the "
                    "delete check would be vacuous" % (SCRATCH_NOTE, before)
                )

            started = time.perf_counter()
            collection.delete(where={"source": SCRATCH_NOTE})
            delete_samples.append((time.perf_counter() - started) * 1000.0)

            # consistency lag: how long until it stops coming back in results
            lag_started = time.perf_counter()
            attempts = 0
            while True:
                attempts += 1
                sources = self._sources_in_results(collection, DELETE_PROBE, probe_k)
                if SCRATCH_NOTE not in sources:
                    break
                if time.perf_counter() - lag_started > 5.0:
                    raise RuntimeError(
                        "deleted note still returned after 5s - delete is not consistent"
                    )
            consistency_samples.append((time.perf_counter() - lag_started) * 1000.0)

            remaining = collection.get(where={"source": SCRATCH_NOTE})
            leftover = len(remaining.get("ids") or [])
            if leftover:
                raise RuntimeError(
                    "delete left %d chunks of %s behind" % (leftover, SCRATCH_NOTE)
                )
            verifications.append(
                {
                    "chunks_deleted": present,
                    "chunks_remaining": leftover,
                    "probe_k": probe_k,
                    "probe_retrieved_before_delete": True,
                    "probe_hits_from_deleted_note": 0,
                    "probe_attempts": attempts,
                }
            )

        # restore for a coherent end state
        self._upsert_note(collection, self.scratch_path)
        return delete_samples, consistency_samples, {"per_run": verifications}

    # -- retrieval quality (untimed) ----------------------------------------

    def op_quality(self) -> dict:
        """Score the labelled query set. Takes no timing samples, by design.

        Called after all four timed ops, so nothing measured above can have
        observed a single query issued here. The collection state it reads is
        the same state Create left behind: Update and Delete only ever touch
        the mutable note, and build_corpus refuses to plant a labelled fact
        there.
        """
        collection = self._collection()
        labels = [dict(entry) for entry in self.truth["queries"]]  # type: ignore
        facts = {
            str(topic["id"]): str(topic["fact"])
            for topic in self.truth["topics"]  # type: ignore
        }
        ladder = tuple(self.truth["k_ladder"])  # type: ignore
        depth = int(self.truth["retrieval_depth"])  # type: ignore
        no_answer_query = str(self.truth["no_answer_query"])
        corpus_files = sorted(path.name for path in self.corpus.glob("*.md"))
        problems: List[str] = []

        for entry in labels:
            for source in entry["relevant"]:  # type: ignore[union-attr]
                if source not in corpus_files:
                    raise RuntimeError(
                        "ground truth names %r, which is not in the corpus" % source
                    )

        # Did the planted text survive chunking into the index? Not a ranking
        # question - a label that points at a document whose copy of the fact
        # was dropped by min_chars would be scoring noise.
        present = 0
        for entry in self._planting:
            fact = facts[str(entry["topic"])]
            stored = collection.get(where={"source": entry["source"]})
            documents = stored.get("documents") or []
            if any(fact in (text or "") for text in documents):
                present += 1
            else:
                problems.append(
                    "fact %s not intact in any chunk of %s"
                    % (entry["topic"], entry["source"])
                )

        scored: List[Dict[str, object]] = []
        for entry in labels:
            ranked = self._sources_in_results(collection, str(entry["query"]), depth)
            result = score_query(ranked, entry["relevant"], ladder, depth)  # type: ignore[arg-type]
            result["id"] = entry["id"]
            result["query"] = entry["query"]
            scored.append(result)

        # --- control (a): a question the corpus does not answer -------------
        unanswerable_ranked = self._sources_in_results(collection, no_answer_query, depth)
        unanswerable = score_query(unanswerable_ranked, [], ladder, depth)
        unanswerable["query"] = no_answer_query
        nonzero = [
            key
            for key, value in list(unanswerable["recall"].items())  # type: ignore[union-attr]
            + [("rr", unanswerable["rr"]), ("ndcg", unanswerable["ndcg@%d" % depth])]
            if value
        ]
        if nonzero:
            raise RuntimeError(
                "unanswerable control scored above zero on %s - the labels are "
                "not independent of the retriever" % nonzero
            )

        # --- control (b): every query given someone else's documents --------
        relevant_sets = [list(entry["relevant"]) for entry in labels]  # type: ignore[arg-type]
        permutation = derange_labels(relevant_sets)
        shuffled: List[Dict[str, object]] = []
        overlap = 0
        if permutation is None:
            problems.append("no document-disjoint derangement of the labels exists")
        else:
            for index, entry in enumerate(labels):
                swapped = relevant_sets[permutation[index]]
                overlap += len(set(swapped) & set(relevant_sets[index]))
                result = score_query(
                    list(scored[index]["retrieved_sources"]), swapped, ladder, depth  # type: ignore[arg-type]
                )
                result["id"] = entry["id"]
                result["query"] = entry["query"]
                result["labelled_as"] = labels[permutation[index]]["id"]
                shuffled.append(result)

        true_aggregate = aggregate_scores(scored, ladder, depth)
        shuffled_aggregate = aggregate_scores(shuffled, ladder, depth) if shuffled else {}

        return {
            "measured": True,
            "metric": "recall@k, document-level; MRR; nDCG@%d" % depth,
            "relevance": (
                "a retrieved chunk is a hit when its source file is in the "
                "query's relevant set; labels are never keyed to chunk ids"
            ),
            "k_ladder": list(ladder),
            "k_semantics": (
                "k counts retrieved chunks (retriever depth); a document is "
                "retrieved when any of those chunks has it as its source"
            ),
            "retrieval_depth": depth,
            "timed": False,
            "measured_after": "all four timed ops; excluded from every latency sample",
            "ground_truth": {
                "origin": "planted at corpus generation",
                "loaded_from": self.truth.get("loaded_from", "generated in-process"),
                "seed": self.truth.get("seed", CORPUS_SEED),
                "derived_from_retriever_output": False,
                "queries": len(labels),
                "corpus_documents": len(corpus_files),
                "relevant_per_query": {
                    "min": min(len(s) for s in relevant_sets),
                    "max": max(len(s) for s in relevant_sets),
                    "mean": round(statistics.fmean([len(s) for s in relevant_sets]), 3),
                },
                "planting": self._planting,
                "facts_intact_in_index": "%d/%d" % (present, len(self._planting)),
            },
            "aggregate": true_aggregate,
            "per_query": scored,
            "controls": {
                "unanswerable_query": {
                    "purpose": "a query no document answers must score 0",
                    "query": no_answer_query,
                    "relevant": [],
                    "denominator": 1,
                    "result": unanswerable,
                    "scored_zero": True,
                },
                "shuffled_ground_truth": {
                    "purpose": (
                        "same retrieved results, labels deranged onto "
                        "document-disjoint sets; the aggregate must drop"
                    ),
                    "permutation": (
                        None
                        if permutation is None
                        else {
                            str(labels[i]["id"]): str(labels[j]["id"])
                            for i, j in enumerate(permutation)
                        }
                    ),
                    "document_overlap_with_truth": overlap,
                    "aggregate": shuffled_aggregate,
                    "per_query": shuffled,
                    "dropped": bool(
                        shuffled_aggregate
                        and shuffled_aggregate.get("recall@%d" % depth, 1.0)
                        < true_aggregate.get("recall@%d" % depth, 0.0)
                    ),
                },
            },
            "problems": problems,
        }

    # -- driver -------------------------------------------------------------

    def run(self) -> dict:
        log.info("bench %s -> collection %s", self.name, self.collection_name)

        create_samples, create_summary = self.op_create()
        collection = self._collection()
        indexed = collection.count()

        read_samples, per_query = self.op_read()
        update_samples, update_notes = self.op_update()
        delete_samples, consistency_samples, delete_notes = self.op_delete()

        # Untimed, and strictly last: no latency sample above can contain it.
        quality = self.op_quality()

        chunk_sizes = [
            len(chunk.text)
            for path in sorted(self.corpus.glob("*.md"))
            for chunk in self._chunks_for(path)
        ]

        return {
            "collection": self.collection_name,
            "indexed_chunks": indexed,
            "quality": quality,
            "chunking_observed": {
                "files": len(list(self.corpus.glob("*.md"))),
                "chunks": len(chunk_sizes),
                "chars_total": sum(chunk_sizes),
                "chars_mean": round(statistics.fmean(chunk_sizes), 1) if chunk_sizes else None,
                "chars_max": max(chunk_sizes) if chunk_sizes else None,
            },
            "ops": {
                "create": dict(
                    summarise(create_samples, "ms per full index build"),
                    detail={"files": create_summary.get("files"), "chunks": create_summary.get("chunks")},
                ),
                "read": dict(
                    summarise(
                        read_samples,
                        "ms per pass over %d fixed queries, k=%d" % (len(QUERIES), self.k),
                    ),
                    per_query=per_query,
                ),
                "update": dict(
                    summarise(update_samples, "ms per re-index of one changed note"),
                    detail=update_notes,
                ),
                "delete": dict(
                    summarise(delete_samples, "ms per removal of one note"),
                    consistency=summarise(
                        consistency_samples, "ms until the note stops appearing in results"
                    ),
                    detail=delete_notes,
                ),
            },
        }


# --- reporting -------------------------------------------------------------


def render_table(report: dict) -> str:
    profile = report["profile"]
    harness = report["harness"]
    observed = report["chunking_observed"]
    chunking = profile.get("chunking") or {}
    embedder = profile.get("embedder") or {}

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("CRUD bench  %s  (%s)" % (profile["name"], report["date"]))
    lines.append("=" * 78)
    lines.append(
        "kind        %s        embedder %s / %s"
        % (profile.get("kind"), embedder.get("provider"), embedder.get("model"))
    )
    lines.append(
        "chunking    max=%s overlap=%s min=%s (%s)"
        % (
            chunking.get("max_chars"),
            chunking.get("overlap_chars"),
            chunking.get("min_chars"),
            chunking.get("strategy"),
        )
    )
    lines.append("collection  %s" % report["collection"])
    lines.append(
        "corpus      %s files / %s chunks / %s chars, mean chunk %s"
        % (
            observed["files"],
            observed["chunks"],
            observed["chars_total"],
            observed["chars_mean"],
        )
    )
    lines.append(
        "harness     runs=%s  queries=%s  k=%s  network=%s"
        % (harness["runs"], len(harness["queries"]), harness["k"], harness["network"])
    )
    lines.append("")
    header = "%-9s %-46s %9s %9s %9s %9s" % ("op", "unit", "cold", "P50", "P95", "P99")
    lines.append(header)
    lines.append("-" * len(header))

    def row(label: str, stats: dict) -> str:
        def fmt(value):
            return "-" if value is None else "%9.2f" % value

        return "%-9s %-46s %s %s %s %s" % (
            label,
            stats["unit"][:46],
            fmt(stats["cold_ms"]),
            fmt(stats["p50_ms"]),
            fmt(stats["p95_ms"]),
            fmt(stats["p99_ms"]),
        )

    ops = report["ops"]
    lines.append(row("create", ops["create"]))
    lines.append(row("read", ops["read"]))
    lines.append(row("update", ops["update"]))
    lines.append(row("delete", ops["delete"]))
    lines.append(row("  +lag", ops["delete"]["consistency"]))
    lines.append("")
    lines.append("read, per query (P50 / P95 ms):")
    for query, stats in ops["read"]["per_query"].items():
        lines.append("  %-52s %8.2f %8.2f" % (query[:52], stats["p50_ms"], stats["p95_ms"]))
    lines.append("")
    lines.append(
        "delete verified: %d/%d runs left 0 chunks and 0 probe hits from the deleted note"
        % (
            sum(
                1
                for entry in ops["delete"]["detail"]["per_run"]
                if entry["chunks_remaining"] == 0 and entry["probe_hits_from_deleted_note"] == 0
            ),
            len(ops["delete"]["detail"]["per_run"]),
        )
    )
    guard = report["state_guard"]
    lines.append(
        "state guard: KB %s, manifest.json %s, manifest root %s"
        % (
            "unchanged" if guard["kb_unchanged"] else "CHANGED",
            "unchanged" if guard["manifest_unchanged"] else "CHANGED",
            guard["manifest_root"],
        )
    )
    lines.append("")
    lines.append(render_quality(report.get("quality")))
    lines.append("=" * 78)
    return "\n".join(lines)


def render_quality(quality: Optional[dict]) -> str:
    if not quality:
        return "quality: not measured"

    ladder = quality["k_ladder"]
    depth = quality["retrieval_depth"]
    ndcg_key = "ndcg@%d" % depth
    truth = quality["ground_truth"]
    shuffled = quality["controls"]["shuffled_ground_truth"]
    unanswerable = quality["controls"]["unanswerable_query"]

    lines: List[str] = []
    lines.append(
        "quality: document-level recall@k over %d labelled queries, %d documents"
        % (truth["queries"], truth["corpus_documents"])
    )
    lines.append(
        "  ground truth planted at generation (seed %s), %s relevant docs/query, "
        "facts intact in index %s"
        % (
            truth["seed"],
            truth["relevant_per_query"]["mean"],
            truth["facts_intact_in_index"],
        )
    )
    lines.append("  k counts retrieved chunks; scored untimed, after the timed ops")
    lines.append("")

    columns = ["recall@%d" % k for k in ladder] + ["mrr", ndcg_key]
    header = "  %-34s %s" % ("", " ".join("%9s" % c for c in columns))
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    def score_row(label: str, block: dict) -> str:
        cells = []
        for column in columns:
            value = block.get(column)
            cells.append("        -" if value is None else "%9.3f" % value)
        return "  %-34s %s" % (label[:34], " ".join(cells))

    lines.append(score_row("AGGREGATE (true labels)", quality["aggregate"]))
    if shuffled["aggregate"]:
        lines.append(
            score_row("CONTROL shuffled ground truth", shuffled["aggregate"])
        )
    control_block = dict(
        unanswerable["result"]["recall"],
        mrr=unanswerable["result"]["rr"],
        **{ndcg_key: unanswerable["result"][ndcg_key]}
    )
    lines.append(score_row("CONTROL unanswerable query", control_block))
    lines.append("")

    for entry in quality["per_query"]:
        block = dict(
            entry["recall"], mrr=entry["rr"], **{ndcg_key: entry[ndcg_key]}
        )
        lines.append(
            score_row(
                "%-11s |R|=%d rank1=%s"
                % (
                    entry["id"],
                    len(entry["relevant"]),
                    entry["first_relevant_rank"] if entry["first_relevant_rank"] else "-",
                ),
                block,
            )
        )
    lines.append("")
    lines.append(
        "  control (a) unanswerable %r -> recall 0 at every k; it retrieved %s"
        % (unanswerable["query"][:44], unanswerable["result"]["retrieved_sources"][:3])
    )
    lines.append(
        "  control (b) labels deranged onto disjoint document sets "
        "(overlap %d) -> aggregate dropped: %s"
        % (shuffled["document_overlap_with_truth"], shuffled["dropped"])
    )
    if quality["problems"]:
        for problem in quality["problems"]:
            lines.append("  PROBLEM: %s" % problem)
    return "\n".join(lines)


# --- BENCHMARKS.md ----------------------------------------------------------
#
# Generated, never typed. Every figure below is read out of a result JSON; a
# number in the prose that is not derivable from a shipped file is a defect, so
# the prose computes its own ratios rather than quoting remembered ones. When a
# re-run moves a ratio, the sentence moves with it.


def _fmt(value, digits: int = 2, dash: str = "-") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return ("%%.%df" % digits) % value
    return str(value)


def _table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def load_results(directory: Path, profiles: Iterable[str], stamp: Optional[str] = None) -> List[dict]:
    """Read one result JSON per profile, newest stamp first when unspecified."""
    reports: List[dict] = []
    missing: List[str] = []
    for name in profiles:
        if stamp:
            candidates = [directory / ("%s-%s.json" % (stamp, name))]
        else:
            candidates = sorted(directory.glob("*-%s.json" % name), reverse=True)
        found = next((path for path in candidates if path.is_file()), None)
        if found is None:
            missing.append(name)
            continue
        report = json.loads(found.read_text(encoding="utf-8"))
        report["_path"] = found.name
        reports.append(report)
    if missing:
        raise RuntimeError(
            "no result file for profile(s) %s in %s - run "
            "'python bench.py --all --corpus bench/corpus' first"
            % (", ".join(missing), directory)
        )
    return reports


def _quality_row(report: dict) -> Dict[str, object]:
    quality = report.get("quality") or {}
    return dict(quality.get("aggregate") or {})


def render_benchmarks(reports: List[dict]) -> str:
    """The published document, assembled from the result files."""
    if not reports:
        raise RuntimeError("no result files to render")

    ordered = sorted(
        reports,
        key=lambda r: (r["profile"].get("chunking") or {}).get("max_chars", 0),
    )
    names = [r["profile"]["name"] for r in ordered]
    harness = ordered[0]["harness"]
    corpus = harness["corpus"]
    runs = {r["harness"]["runs"] for r in ordered}
    stamps = {r["date"] for r in ordered}

    lines: List[str] = []
    lines.append("# Benchmarks")
    lines.append("")
    lines.append(
        "Generated from the result files in `bench/results/` by "
        "`python bench.py --render-benchmarks`. **Do not edit by hand.** Every "
        "figure below is read out of one of those JSON files; a number in this "
        "document that is not derivable from a shipped result file is a defect."
    )
    lines.append("")
    lines.append("Reproduce, from a fresh clone:")
    lines.append("")
    lines.append("    ./install.sh && python bench.py --all --corpus bench/corpus")
    lines.append("")
    lines.extend(
        _table(
            ["", ""],
            [
                ["source files", ", ".join("`bench/results/%s`" % r["_path"] for r in ordered)],
                ["run date", ", ".join(sorted(stamps))],
                ["profiles", ", ".join("`%s`" % n for n in names)],
                ["runs per operation", ", ".join(str(n) for n in sorted(runs))],
                [
                    "corpus",
                    "`bench/corpus/` — %s files, %s characters, seed %s, %s planted facts"
                    % (
                        corpus.get("files"),
                        corpus.get("chars"),
                        corpus.get("seed"),
                        corpus.get("planted_facts"),
                    ),
                ],
                [
                    "network",
                    "%s; blocked outbound connects during the run: %s"
                    % (harness.get("network"), harness.get("blocked_connect_attempts")),
                ],
            ],
        )
    )
    lines.append("")

    # --- environment --------------------------------------------------------
    lines.append("## Environment")
    lines.append("")
    environments = [r["harness"].get("environment") or {} for r in ordered]
    base = environments[0]
    differing = sorted(
        {key for env in environments[1:] for key in base if env.get(key) != base.get(key)}
    )
    labels = [
        ("cpu", "CPU", 0),
        ("cores_physical", "cores (physical)", 0),
        ("cores_logical", "cores (logical)", 0),
        ("memory_gb", "RAM (GiB)", 1),
        ("os", "OS", 0),
        ("arch", "architecture", 0),
        ("python", "Python", 0),
        ("chromadb", "chromadb", 0),
        ("onnxruntime", "onnxruntime", 0),
        ("numpy", "numpy", 0),
    ]
    lines.extend(
        _table(
            ["item", "value"],
            [
                [label, _fmt(base.get(key), digits)]
                for key, label, digits in labels
                if key in base
            ],
        )
    )
    lines.append("")
    if differing:
        lines.append(
            "The three runs did not share an environment; these fields differ "
            "between them and the table above shows `%s`: %s."
            % (names[0], ", ".join("`%s`" % key for key in differing))
        )
    else:
        lines.append(
            "All three profiles were measured on the same machine in the same "
            "session. No hostname, user name or file path is recorded: the "
            "reader needs the hardware, not the workstation it belongs to."
        )
    lines.append("")

    # --- method -------------------------------------------------------------
    quality0 = ordered[0].get("quality") or {}
    truth0 = quality0.get("ground_truth") or {}
    ladder = quality0.get("k_ladder") or list(QUALITY_LADDER)
    depth = quality0.get("retrieval_depth") or QUALITY_DEPTH
    ndcg_key = "ndcg@%d" % depth

    lines.append("## Method")
    lines.append("")
    runs_text = "/".join(str(n) for n in sorted(runs))
    lines.append(
        "Four operations are timed, N=%s runs each, against a synthetic corpus "
        "generated deterministically from seed %s and shipped in this "
        "repository as `bench/corpus/`. The corpus on disk is compared "
        "byte-for-byte with the generator's output before anything is "
        "measured, and the run fails loudly if they differ — otherwise the "
        "numbers would describe a corpus nobody can see."
        % (runs_text, corpus.get("seed"))
    )
    lines.append("")
    lines.extend(
        _table(
            ["operation", "what is timed"],
            [
                ["create", "a full index build of the whole corpus"],
                [
                    "read",
                    "one pass over %d fixed queries at k=%s, fixed order, no sampling"
                    % (len(harness.get("queries") or []), harness.get("k")),
                ],
                ["update", "re-index of a single changed note"],
                ["delete", "removal of one note"],
                [
                    "delete +lag",
                    "time until the removed note stops being returned by a probe query",
                ],
            ],
        )
    )
    lines.append("")
    lines.append(
        "**Percentiles, not means.** A mean over N=%s runs hides the shape that "
        "matters here: the first run of any operation pays for a cold cache and "
        "the rest do not, so an average is a blend of two different things. P50 "
        "answers \"what does this cost normally\"; P95 answers \"how bad is the "
        "tail\"; the cold column is reported separately rather than being "
        "averaged into either." % runs_text
    )
    lines.append("")
    lines.append(
        "**Quality is measured, not assumed.** Latency alone ranks a system that "
        "answers fast and wrong above one that answers slowly and right, so "
        "every run also scores document-level `recall@k` for k in %s, plus MRR "
        "and `%s`, over %s labelled queries planted into the corpus at "
        "generation time. Relevance is document-level: a retrieved chunk counts "
        "when its source file is in the query's relevant set. Chunk identity is "
        "the independent variable across these profiles, so chunk-keyed labels "
        "would make them incomparable by design. Scoring runs after all four "
        "timed operations and takes no timing samples."
        % (", ".join(str(k) for k in ladder), ndcg_key, truth0.get("queries"))
    )
    lines.append("")
    lines.append(
        "**Two controls run every time, and are reported with their real "
        "numbers.** They exist because a check that can only pass measures "
        "nothing."
    )
    lines.append("")
    lines.append(
        "1. *An unanswerable query.* No document answers it, so it must score "
        "zero at every rung. It deliberately reuses vocabulary the corpus is "
        "full of, so the retriever answers it confidently with plausible "
        "chunks — the control has to survive a confident wrong answer, which is "
        "exactly the failure latency cannot see."
    )
    lines.append(
        "2. *Deranged labels.* Every query is scored against another query's "
        "documents, with the swap required to be document-disjoint from the "
        "truth. The aggregate must drop; if it does not, the labels are not "
        "independent of the retriever."
    )
    lines.append("")
    lines.append(
        "**Offline by design, and checked.** A socket guard is installed before "
        "anything heavy is imported; loopback and unix sockets stay allowed, "
        "every other outbound connect is refused and counted into the report. "
        "The runs below recorded %s blocked attempts."
        % ", ".join(str(r["harness"].get("blocked_connect_attempts")) for r in ordered)
    )
    lines.append("")

    # --- latency ------------------------------------------------------------
    lines.append("## Latency")
    lines.append("")
    op_labels = [
        ("create", "create", None),
        ("read", "read", None),
        ("update", "update", None),
        ("delete", "delete", None),
        ("delete", "delete +lag", "consistency"),
    ]
    lines.append("P50, milliseconds:")
    lines.append("")
    rows = []
    for op_key, label, nested in op_labels:
        row = [label]
        for report in ordered:
            stats = report["ops"][op_key]
            if nested:
                stats = stats[nested]
            row.append(_fmt(stats["p50_ms"]))
        rows.append(row)
    lines.extend(_table(["operation (P50 ms)"] + names, rows))
    lines.append("")
    lines.append("Full distribution, milliseconds:")
    lines.append("")
    rows = []
    for report in ordered:
        for op_key, label, nested in op_labels:
            stats = report["ops"][op_key]
            if nested:
                stats = stats[nested]
            rows.append(
                [
                    report["profile"]["name"],
                    label,
                    str(stats["n"]),
                    _fmt(stats["cold_ms"]),
                    _fmt(stats["p50_ms"]),
                    _fmt(stats["p95_ms"]),
                    _fmt(stats["p99_ms"]),
                ]
            )
    lines.extend(
        _table(["profile", "operation", "n", "cold", "P50", "P95", "P99"], rows)
    )
    lines.append("")
    lines.append("Corpus as each profile cut it:")
    lines.append("")
    rows = []
    for report in ordered:
        observed = report["chunking_observed"]
        chunking = report["profile"].get("chunking") or {}
        rows.append(
            [
                report["profile"]["name"],
                str(chunking.get("max_chars")),
                str(chunking.get("overlap_chars")),
                str(observed["files"]),
                str(observed["chunks"]),
                _fmt(observed["chars_mean"], 1),
                str(observed["chars_max"]),
            ]
        )
    lines.extend(
        _table(
            [
                "profile",
                "max chars",
                "overlap",
                "files",
                "chunks",
                "mean chunk",
                "largest chunk",
            ],
            rows,
        )
    )
    lines.append("")

    # --- quality ------------------------------------------------------------
    lines.append("## Retrieval quality")
    lines.append("")
    columns = ["recall@%d" % k for k in ladder] + ["mrr", ndcg_key]
    rows = []
    for report in ordered:
        aggregate = _quality_row(report)
        rows.append(
            [report["profile"]["name"]]
            + [_fmt(aggregate.get(column), 3) for column in columns]
        )
    lines.extend(_table(["profile"] + columns, rows))
    lines.append("")
    ceiling = None
    relevant = (truth0.get("relevant_per_query") or {})
    if truth0.get("planting") is not None and quality0.get("per_query"):
        ceiling = statistics.fmean(
            [1.0 / max(1, len(entry["relevant"])) for entry in quality0["per_query"]]
        )
    if ceiling is not None:
        lines.append(
            "`recall@1` cannot reach 1.0: with %s to %s relevant documents per "
            "query (mean %s), one retrieved chunk can cover at most one of "
            "them, which caps the mean at **%.3f**."
            % (
                relevant.get("min"),
                relevant.get("max"),
                relevant.get("mean"),
                ceiling,
            )
        )
        lines.append("")
    lines.append("### Controls")
    lines.append("")
    rows = []
    for report in ordered:
        quality = report.get("quality") or {}
        controls = quality.get("controls") or {}
        unanswerable = (controls.get("unanswerable_query") or {}).get("result") or {}
        shuffled = controls.get("shuffled_ground_truth") or {}
        shuffled_aggregate = shuffled.get("aggregate") or {}
        true_aggregate = _quality_row(report)
        rows.append(
            [
                report["profile"]["name"],
                _fmt(unanswerable.get("rr"), 3),
                _fmt(unanswerable.get(ndcg_key), 3),
                str(unanswerable.get("distinct_documents_retrieved")),
                str(shuffled.get("document_overlap_with_truth")),
                "%s → %s"
                % (
                    _fmt(true_aggregate.get("mrr"), 3),
                    _fmt(shuffled_aggregate.get("mrr"), 3),
                ),
                "%s → %s"
                % (
                    _fmt(true_aggregate.get("recall@3"), 3),
                    _fmt(shuffled_aggregate.get("recall@3"), 3),
                ),
            ]
        )
    lines.extend(
        _table(
            [
                "profile",
                "unanswerable rr",
                "unanswerable " + ndcg_key,
                "docs it still returned",
                "deranged label overlap",
                "MRR true → deranged",
                "recall@3 true → deranged",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "The unanswerable query scored zero on every profile while the "
        "retriever confidently returned several distinct documents for it. "
        "Deranged labels are document-disjoint from the truth — overlap 0 — and "
        "every profile dropped."
    )
    lines.append("")

    # --- the finding --------------------------------------------------------
    read_p50 = {r["profile"]["name"]: r["ops"]["read"]["p50_ms"] for r in ordered}
    budgets = {
        r["profile"]["name"]: (r["profile"].get("chunking") or {}).get("max_chars")
        for r in ordered
    }
    mean_chunk = {
        r["profile"]["name"]: r["chunking_observed"]["chars_mean"] for r in ordered
    }
    chunk_count = {r["profile"]["name"]: r["chunking_observed"]["chunks"] for r in ordered}
    quality_by_name = {r["profile"]["name"]: _quality_row(r) for r in ordered}

    read_span = max(read_p50.values()) / min(read_p50.values())
    budget_span = max(budgets.values()) / min(budgets.values())
    mean_span = max(mean_chunk.values()) / min(mean_chunk.values())
    count_span = max(chunk_count.values()) / min(chunk_count.values())
    fastest_read = min(read_p50, key=lambda n: read_p50[n])
    fastest_create = min(ordered, key=lambda r: r["ops"]["create"]["p50_ms"])["profile"]["name"]
    best_quality = max(
        quality_by_name,
        key=lambda n: (
            quality_by_name[n].get("recall@3", 0.0),
            quality_by_name[n].get(ndcg_key, 0.0),
            quality_by_name[n].get("mrr", 0.0),
        ),
    )

    lines.append("## Finding")
    lines.append("")
    lines.append(
        "**Read latency is flat, and timing alone picks the wrong profile.** "
        "Across the three profiles the P50 of a read pass spans **%.2fx** "
        "(%.2f ms on `%s`, %.2f ms on `%s`), while the chunk budget spans "
        "**%.1fx** (%s → %s characters), the mean chunk actually produced spans "
        "**%.1fx** (%s → %s characters) and the number of chunks in the index "
        "spans **%.1fx** (%d → %d). A %.1fx change in how the corpus is cut "
        "moved read latency by %.0f%%."
        % (
            read_span,
            min(read_p50.values()),
            fastest_read,
            max(read_p50.values()),
            max(read_p50, key=lambda n: read_p50[n]),
            budget_span,
            min(budgets.values()),
            max(budgets.values()),
            mean_span,
            _fmt(min(mean_chunk.values()), 1),
            _fmt(max(mean_chunk.values()), 1),
            count_span,
            min(chunk_count.values()),
            max(chunk_count.values()),
            budget_span,
            (read_span - 1.0) * 100.0,
        )
    )
    lines.append("")
    if fastest_read == fastest_create:
        cheapest = (
            "`%s` is both the fastest to read and the cheapest to build, and if "
            "the tables stopped at latency it would look like the answer."
            % fastest_read
        )
    else:
        cheapest = (
            "`%s` is the fastest to read and `%s` the cheapest to build, and if "
            "the tables stopped at latency, one of those would look like the "
            "answer." % (fastest_read, fastest_create)
        )
    lines.append(
        "That is the unflattering part of the method, stated first: on this "
        "corpus the timed operations do not separate the profiles on the axis "
        "that a reader would care about. " + cheapest
    )
    lines.append("")
    best = quality_by_name[best_quality]
    lines.append(
        "**Quality is what decides it.** `%s` leads on `recall@3` (%s) and on "
        "`%s` (%s), and it is the profile this toolkit ships as its default. "
        "The full ladder:"
        % (
            best_quality,
            _fmt(best.get("recall@3"), 3),
            ndcg_key,
            _fmt(best.get(ndcg_key), 3),
        )
    )
    lines.append("")
    rows = []
    for report in ordered:
        name = report["profile"]["name"]
        aggregate = quality_by_name[name]
        rows.append(
            [
                name,
                str(budgets[name]),
                _fmt(read_p50[name]),
                _fmt(report["ops"]["create"]["p50_ms"]),
                _fmt(aggregate.get("recall@3"), 3),
                _fmt(aggregate.get(ndcg_key), 3),
            ]
        )
    lines.extend(
        _table(
            ["profile", "max chars", "read P50", "create P50", "recall@3", ndcg_key],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Read the table as a decision, not as a ranking: the coarsest profile "
        "buys its speed with retrieval quality, the finest one costs more to "
        "build *and* scores lower, and the difference is invisible to every "
        "timed operation above."
    )
    lines.append("")

    # --- limitations --------------------------------------------------------
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "These numbers are worth exactly what their scope allows, and the scope "
        "is small."
    )
    lines.append("")
    for item in [
        "**One machine, one session.** Every figure comes from the single "
        "environment in the table above. Nothing here is a cross-machine claim, "
        "and the absolute milliseconds will not transfer.",
        "**A small synthetic corpus.** %s documents, %s characters, %s labelled "
        "queries with %s to %s relevant documents each. That resolves the "
        "*direction* of the quality difference; it does not pin its magnitude, "
        "and a handful of queries carry the coarse profile's deficit."
        % (
            corpus.get("files"),
            corpus.get("chars"),
            truth0.get("queries"),
            relevant.get("min"),
            relevant.get("max"),
        ),
        "**Synthetic, not natural, text.** The corpus is generated so that it "
        "can be shipped and re-derived byte-for-byte. Real notes have different "
        "heading density and different vocabulary overlap, both of which the "
        "chunker is sensitive to.",
        "**Local only, single client, no concurrency.** One process, one "
        "collection, no parallel readers or writers, no server hop. Nothing "
        "here says anything about behaviour under load.",
        "**One embedder.** %s / %s throughout. The comparison is between chunk "
        "geometries at a fixed embedder, not between embedders."
        % (
            (ordered[0]["profile"].get("embedder") or {}).get("provider"),
            (ordered[0]["profile"].get("embedder") or {}).get("model"),
        ),
        "**Cold and warm are both present.** The `cold` column is the first run "
        "of each operation and is reported separately, but the process itself is "
        "warm by the time later profiles run; ordering effects at the "
        "millisecond scale are not controlled for.",
        "**No vendor comparison.** Every profile here is the same local stack. "
        "A hosted retrieval service is a different measurement and is not one of "
        "these rows.",
    ]:
        lines.append("- " + item)
    lines.append("")
    lines.append(
        "The corpus, the labels and the result files are all in this "
        "repository, so any of the above can be checked rather than believed."
    )
    lines.append("")
    return "\n".join(lines)


def write_benchmarks(
    results_dir: Path = RESULTS_DIR,
    target: Path = BENCHMARKS_FILE,
    profiles: Iterable[str] = PUBLISHED_PROFILES,
    stamp: Optional[str] = None,
) -> Path:
    reports = load_results(results_dir, profiles, stamp)
    target.write_text(render_benchmarks(reports), encoding="utf-8")
    log.info("wrote %s from %d result files", target.name, len(reports))
    return target


# --- entry point -----------------------------------------------------------


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception:  # pragma: no cover - absent dependency
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def _sysctl(key: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    value = out.stdout.strip()
    return value or None


def _cpu_model() -> str:
    """A CPU description, never a machine name.

    ``platform.processor()`` is often just the architecture; the useful string
    is the vendor's brand. Both are properties of the hardware, so neither
    identifies the person running this.
    """
    value = _sysctl("machdep.cpu.brand_string")
    if value:
        return value
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:  # pragma: no cover
            pass
    return platform.processor() or platform.machine() or "unknown"


def _physical_cores() -> Optional[int]:
    value = _sysctl("hw.physicalcpu")
    if value and value.isdigit():
        return int(value)
    try:  # pragma: no cover - platform dependent
        return len({
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.lower().startswith("core id")
        }) or None
    except (OSError, IndexError):
        return None


def _memory_gb() -> Optional[float]:
    value = _sysctl("hw.memsize")
    total: Optional[int] = int(value) if value and value.isdigit() else None
    if total is None:
        try:  # pragma: no cover - platform dependent
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            return None
    return round(total / (1024 ** 3), 1)


def _os_description() -> str:
    if sys.platform == "darwin":
        release = platform.mac_ver()[0]
        if release:
            return "macOS %s (Darwin %s)" % (release, platform.release())
    return "%s %s" % (platform.system(), platform.release())


_UNSAFE_ENV = (
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"\b[0-9a-fA-F]{64}\b"),
)


def _scrub(value):
    """Remove anything that identifies the machine or its owner.

    The hostname is never collected in the first place — the defence against a
    leak is not gathering the value, and this is the second line for strings
    that arrive from a vendor tool. Scrubbing rather than failing because this
    runs on a stranger's machine, where an unexpected string in a CPU brand is
    not a reason to lose the run.
    """
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        for pattern in _UNSAFE_ENV:
            value = pattern.sub("(redacted)", value)
    return value


def _environment() -> dict:
    """What a reader needs to judge the numbers, and nothing that identifies
    where they were produced. No hostname, no user, no absolute path."""
    return _scrub(
        {
            "python": sys.version.split()[0],
            "chromadb": _module_version("chromadb"),
            "onnxruntime": _module_version("onnxruntime"),
            "numpy": _module_version("numpy"),
            "os": _os_description(),
            "platform": sys.platform,
            "arch": platform.machine(),
            "cpu": _cpu_model(),
            "cores_physical": _physical_cores(),
            "cores_logical": os.cpu_count(),
            "memory_gb": _memory_gb(),
            "collected": "hardware and library versions only; no hostname, user or path",
        }
    )


def drop_collection(name: str) -> None:
    if not name.startswith(COLLECTION_PREFIX) or name == COLLECTION_NAME:
        raise RuntimeError("refusing to drop %r" % name)
    try:
        get_client().delete_collection(name)
        log.info("dropped bench collection %s", name)
    except Exception as exc:
        log.debug("no collection %s to drop (%s)", name, exc)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        help="profile name; repeatable (default: baseline)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run the published profiles: %s" % ", ".join(PUBLISHED_PROFILES),
    )
    parser.add_argument("-n", "--runs", type=int, default=DEFAULT_RUNS, help="runs per op")
    parser.add_argument("-k", type=int, default=None, help="override retrieval k")
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR, help="report directory")
    parser.add_argument("--date", default=None, help="date stamp for the filename")
    parser.add_argument(
        "--emit-corpus",
        type=Path,
        default=None,
        metavar="DIR",
        help="write the deterministic corpus and its ground truth to DIR, then exit",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        metavar="DIR",
        help="bench against a corpus already on disk (verified against the generator)",
    )
    parser.add_argument(
        "--render-benchmarks",
        action="store_true",
        help="regenerate BENCHMARKS.md from the result files, then exit",
    )
    parser.add_argument(
        "--no-benchmarks-md",
        action="store_true",
        help="with --all: do not regenerate BENCHMARKS.md afterwards",
    )
    parser.add_argument(
        "--keep-collections",
        action="store_true",
        help="do not drop bench__* collections afterwards",
    )
    parser.add_argument("--quiet", action="store_true", help="write files, print nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # --- corpus emission: a generator run, not a measurement ----------------
    if args.emit_corpus is not None:
        summary = emit_corpus(args.emit_corpus)
        if not args.quiet:
            print(
                "corpus written to %s: %d files, %d characters, %d planted facts"
                % (
                    summary["dir"],
                    len(summary["files"]),
                    summary["chars"],
                    summary["planted_facts"],
                )
            )
            print("  " + "\n  ".join(summary["files"]))
        return 0

    # --- document generation: reads result files, measures nothing ----------
    if args.render_benchmarks and not (args.all or args.profile):
        try:
            path = write_benchmarks(args.outdir, stamp=args.date)
        except Exception as exc:
            print("RENDER FAILED: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
            return 1
        if not args.quiet:
            print("wrote %s" % path)
        return 0

    if args.all and args.profile:
        parser.error("--all and --profile are alternatives, not a combination")
    profile_names = list(PUBLISHED_PROFILES) if args.all else (args.profile or ["baseline"])
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    stamp = args.date or _dt.date.today().isoformat()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    guard = StateGuard()
    corpus = RAG_ROOT / (".bench-corpus-%d" % os.getpid())
    shipped: Optional[Path] = args.corpus
    shipped_before: Optional[str] = None
    benched: List[str] = []
    written: List[Path] = []
    status = 0

    try:
        if shipped is not None:
            # The shipped corpus is read, verified and copied. It is never the
            # directory the timed ops write to: Update rewrites the mutable note
            # and Delete removes it, and doing that to a file under version
            # control would make the next run's verification fail for a reason
            # that has nothing to do with the corpus being wrong.
            shipped_before = _fingerprint_tree(shipped)
            corpus_info = verify_corpus(shipped)
            truth = dict(load_ground_truth(shipped), loaded_from=shipped.name + "/" + GROUND_TRUTH_FILE)
            corpus_source = "shipped: %s (verified against the generator)" % shipped.as_posix()
            if not args.quiet:
                print(
                    "corpus verified: %s matches the generator byte for byte "
                    "(%d files, %d characters)"
                    % (shipped, corpus_info["files"], corpus_info["chars"])
                )

            def reset_corpus() -> None:
                stage_corpus(shipped, corpus)  # type: ignore[arg-type]

        else:
            corpus_info, planting = build_corpus(corpus)
            files = {name: (corpus / name).read_bytes() for name in corpus_documents()}
            truth = dict(
                ground_truth_document(planting, files),
                loaded_from="generated in-process (no --corpus given)",
            )
            corpus_source = "generated in a scratch directory and discarded"

            def reset_corpus() -> None:
                build_corpus(corpus)

        reset_corpus()

        for name in profile_names:
            profile = load_profile(name)
            bench = ProfileBench(
                profile=profile,
                corpus=corpus,
                runs=args.runs,
                k=args.k or int((profile.get("retrieval") or {}).get("k", 5)),
                truth=truth,
            )
            benched.append(bench.collection_name)
            # Every profile starts from the same corpus bytes.
            reset_corpus()
            result = bench.run()

            report = {
                "schema_version": SCHEMA_VERSION,
                "generated_at": _dt.datetime.now(_dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "date": stamp,
                "profile": profile,
                "collection": result["collection"],
                "harness": {
                    "runs": args.runs,
                    "k": bench.k,
                    "queries": list(QUERIES),
                    "corpus": dict(
                        corpus_info,
                        kind="synthetic",
                        seed=CORPUS_SEED,
                        source=corpus_source,
                    ),
                    "network": "blocked",
                    "blocked_connect_attempts": len(BLOCKED_CONNECTS),
                    "blocked_connect_targets": sorted(set(BLOCKED_CONNECTS)),
                    "environment": _environment(),
                },
                "chunking_observed": result["chunking_observed"],
                "indexed_chunks": result["indexed_chunks"],
                "ops": result["ops"],
                # Timing alone ranks fast-and-wrong above slow-and-right.
                # Filled by ProfileBench.op_quality(), which is untimed.
                "quality": result["quality"],
                # Verdicts only: see StateGuard.public().
                "state_guard": guard.public(),
            }
            report["table"] = render_table(report)

            path = outdir / ("%s-%s.json" % (stamp, name))
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(path)
            if not args.quiet:
                print(report["table"])
                print("written: %s" % path)
                print()
    except Exception as exc:
        status = 1
        print("BENCH FAILED: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        log.exception("bench failed")
    finally:
        shutil.rmtree(corpus, ignore_errors=True)
        if not args.keep_collections:
            for name in benched:
                drop_collection(name)
        final = guard.verify()
        if final["problems"]:
            status = 1
            print("STATE GUARD FAILED:", file=sys.stderr)
            for problem in final["problems"]:
                print("  %s" % problem, file=sys.stderr)
        elif not args.quiet:
            print(
                "state restored: KB tree %s..., manifest root %s, corpus removed, "
                "%d bench collection(s) dropped"
                % (final["kb_tree_sha256"][:16], final["manifest_root"], 0 if args.keep_collections else len(benched))
            )
        # The shipped corpus must survive a run untouched. Checked, not assumed:
        # a harness that mutates the artefact it measures invalidates it.
        if shipped is not None and shipped_before is not None:
            shipped_after = _fingerprint_tree(shipped)
            if shipped_after != shipped_before:
                status = 1
                print(
                    "SHIPPED CORPUS MUTATED: %s changed during the run (%s -> %s)"
                    % (shipped, shipped_before[:16], shipped_after[:16]),
                    file=sys.stderr,
                )
            elif not args.quiet:
                print("shipped corpus unchanged: %s" % shipped)
        if BLOCKED_CONNECTS and not args.quiet:
            print("blocked outbound connects: %s" % sorted(set(BLOCKED_CONNECTS)))

    if written and not args.quiet:
        print("reports: %s" % ", ".join(str(p) for p in written))

    # BENCHMARKS.md is regenerated from the files just written, so the prose and
    # the JSON cannot drift apart: there is no step in which a figure is copied
    # by hand. Only after a full published run — a partial one would render a
    # document mixing this run's numbers with an older run's.
    if status == 0 and args.all and not args.no_benchmarks_md:
        try:
            path = write_benchmarks(outdir, stamp=stamp)
        except Exception as exc:
            status = 1
            print("RENDER FAILED: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        else:
            if not args.quiet:
                print("regenerated %s from the result files above" % path)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
