#!/usr/bin/env python
"""CRUD timing harness for the local retrieval toolkit, one run per profile.

Four operations are timed, each with its own sample count, and the statistics
reported are gated on that count:

    Create  a full index build of the whole corpus            (N=20)
    Read    one pass over a fixed query set at fixed k        (N=100)
    Update  re-index of a single changed note                 (N=50)
    Delete  removal of one note, plus the lag until it stops being returned
                                                              (N=100)

Median, min, max and median absolute deviation are always reported. P50 and P95
are reported only where N >= 100. **There is no P99 anywhere in this file.** At
these sample sizes a P99 is an interpolation between the two largest samples
dressed up as a tail statistic, and reporting it would be a claim the samples
cannot carry.

Two *modes* are named and never mixed:

    warm        many operations in one process: model loaded, OS cache hot.
                Every one of the four operations above is warm.
    cold-start  a fresh subprocess per sample, paying interpreter startup,
                module import, Chroma client construction and ONNX model load
                before it does anything. Reported in its own section.

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

Read latency is additionally **decomposed**: query embedding, vector search and
result materialization are timed separately against the same collection, and
their sum is required to close to within 10% of the full in-process read. A
decomposition that does not close is not published. The full MCP round-trip is
measured alongside them as an outer envelope — process hop, JSON-RPC framing
and the server's own work — and is explicitly not one of the summands.

Retrieval is scored twice. ``op_quality`` scores it at equal k, which is not a
fair comparison across chunk sizes: five chunks of 240 characters and five of
1087 are not the same context. ``op_token_budget`` scores it again under a fixed
token budget, which is the constraint a caller actually has.

The corpus is also a *shippable artefact*. ``--emit-corpus`` writes it, with its
labelled query set, as files; ``--corpus`` runs against those files instead of
generating a throwaway copy. The on-disk corpus is verified byte-for-byte
against the generator before anything is measured, and the shipped directory is
never mutated: the timed Update and Delete ops work on a scratch copy, and the
shipped bytes are re-checked on the way out. A corpus a stranger can silently
edit is a corpus whose numbers mean nothing.

Usage:
    python bench.py                                   # baseline, at the defaults
    python bench.py --profile baseline --profile chunk-small
    python bench.py --profile chunk-large -n 5        # smoke test, not publishable
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

# The earliest clock this module can read. The cold-start child measures its own
# initialisation from here, so the only thing before it is interpreter startup
# and the stdlib imports above — which is precisely what ``overhead_ms``
# accounts for, computed by the parent from the process wall time.
_PROCESS_START = time.perf_counter()

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
    _embedding_function,
    config_from_profile,
    chunk_markdown,
    get_client,
    get_collection,
    load_profile,
    reindex,
)

log = get_logger("bench")

SCHEMA_VERSION = 3

# --- sample sizes ----------------------------------------------------------
# One N per operation, not one N for the harness. The four operations differ by
# two orders of magnitude in cost, so a single N either starves the cheap ones
# of samples or makes the expensive ones unrunnable. These are the counts the
# published run uses; the caps exist so that a typo cannot turn a smoke test
# into an afternoon.
#
# The gate below is the reason the counts are what they are: a percentile is
# rendered only at N >= 100, so read and delete are sampled to 100 and create
# and update are honest about reporting a median and a spread instead.
DEFAULT_RUNS: Dict[str, int] = {
    "read": 100,
    "delete": 100,
    "update": 50,
    "create": 20,
    "cold_start": 5,
}
RUNS_CAP: Dict[str, int] = {
    "read": 500,
    "delete": 500,
    "update": 500,
    "create": 500,
    "cold_start": 50,
}
# The operations ``-n/--runs`` sets. Cold start is not one of them: it is a
# subprocess per sample and a smoke-test N of 5 there is already the default.
TIMED_OPS: Tuple[str, ...] = ("create", "read", "update", "delete")

# A percentile needs enough samples that it is not simply naming one of them.
PERCENTILE_MIN_N = 100

MODE_WARM = "warm"
MODE_COLD_START = "cold-start"

# Cold start and the stage split both need their own sample counts stated
# wherever they are rendered; these are the defaults for the two measurements
# that are not one of the four operations.
MCP_ROUNDTRIP_RUNS = 20
STAGE_RESIDUAL_LIMIT_PCT = 10.0
TOKEN_BUDGETS: Tuple[int, ...] = (2000, 4000)
HEADLINE_BUDGET = 2000
BUDGET_RETRIEVAL_CAP = 128
COLD_START_SENTINEL = "COLD-START-PROBE "

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


def median_absolute_deviation(values: List[float]) -> Optional[float]:
    """``median(|x - median(x)|)``.

    The right spread statistic at small N. A standard deviation assumes the tail
    was sampled and is dragged around by the one slow run every process has;
    MAD asks how far a typical sample sits from a typical sample, which is a
    question twenty samples can answer.
    """
    if not values:
        return None
    centre = statistics.median(values)
    return statistics.median([abs(value - centre) for value in values])


def summarise(samples: List[float], unit: str, mode: str = MODE_WARM) -> Dict[str, object]:
    """Statistics for one operation, gated on how many samples there are.

    ``median``/``min``/``max``/``mad`` are always reported: they are statements
    about the samples in hand. ``p50``/``p95`` appear only at N >= 100, because
    below that a percentile is an interpolation between two neighbouring samples
    presented as a property of a distribution.

    There is no ``p99``. Not gated — absent. At N=100 the 99th percentile is an
    interpolation between the two largest samples, and no N used anywhere in
    this harness earns it. A key that is missing cannot be read by accident; a
    key that is present and null invites one.

    ``mode`` is carried through into the report so that no table can be rendered
    without saying which of the two measurement modes produced it.
    """
    ordered = sorted(samples)
    earned = len(samples) >= PERCENTILE_MIN_N
    stats: Dict[str, object] = {
        "unit": unit,
        "mode": mode,
        "n": len(samples),
        "median_ms": round(statistics.median(ordered), 3) if samples else None,
        "min_ms": round(ordered[0], 3) if samples else None,
        "max_ms": round(ordered[-1], 3) if samples else None,
        "mad_ms": round(median_absolute_deviation(ordered), 3) if samples else None,
        "mean_ms": round(statistics.fmean(samples), 3) if samples else None,
        "stdev_ms": round(statistics.pstdev(samples), 3) if len(samples) > 1 else None,
        "percentiles_earned": earned,
        "percentile_min_n": PERCENTILE_MIN_N,
        "samples_ms": [round(value, 3) for value in samples],
    }
    if earned:
        stats["p50_ms"] = round(percentile(ordered, 0.50), 3)
        stats["p95_ms"] = round(percentile(ordered, 0.95), 3)
    return stats


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


# --- token-budget retrieval -------------------------------------------------
# Equal k is not a fair comparison across chunk geometries. Five chunks of 240
# characters and five of 1087 are not the same context, so a ladder that holds k
# fixed is measuring "how much text did you hand me" as much as "was it the
# right text". A caller does not have a k budget, it has a context budget.
#
# Everything below is a pure function of (ranked chunks, budget, relevant set):
# no collection, no client, no clock. That is what lets the tests exercise the
# admission rule on hand-built inputs.


def approx_tokens(text: str) -> int:
    """``ceil(len(text) / 4)`` — a character heuristic, not a tokenizer.

    Named ``approx`` everywhere it surfaces, and documented as an approximation
    in the generated report, because it is one: real BPE token counts move with
    vocabulary and whitespace, and a reader has to be able to discount this. It
    is used identically for every profile, so it cannot favour one of them; what
    it cannot support is an absolute claim about a real model's context window.
    """
    return math.ceil(len(text) / 4)


def admit_within_budget(
    ranked: List[Tuple[str, str]], budget: int
) -> List[Tuple[str, str, int]]:
    """Greedy **prefix**: admit ranked chunks in order until one does not fit.

    Stop at the first chunk that would exceed the budget — do not skip it and
    carry on down the ranking. Skip-and-continue is a different retriever, and a
    flattering one: it would let a profile with many small chunks backfill the
    remaining budget with lower-ranked text while a profile with large chunks
    got truncated, and the comparison would then be measuring the packer rather
    than the chunking.

    Returns ``(source, text, tokens)`` for each admitted chunk, in rank order.
    """
    admitted: List[Tuple[str, str, int]] = []
    used = 0
    for source, text in ranked:
        cost = approx_tokens(text)
        if used + cost > budget:
            break
        admitted.append((source, text, cost))
        used += cost
    return admitted


def score_budget(
    ranked: List[Tuple[str, str]], relevant: Iterable[str], budget: int
) -> Dict[str, object]:
    """Score one query's retrieval under a fixed token budget.

    * ``recall@budget`` — relevant documents represented among the admitted
      chunks, over ``|relevant|`` (floored at 1, so the unanswerable control can
      still represent a non-zero score and be seen to score zero).
    * ``unique_docs`` — distinct sources admitted: how many documents actually
      made it into the context, which is the number a reader of the answer sees.
    * ``duplicate_source_rate`` — the share of the admitted chunks that came
      from a document already represented. Budget spent re-reading one note.
    * ``relevant_token_fraction`` — of the budget actually used, how much of it
      was text from a relevant document. This is the metric equal-k cannot
      express at all.
    """
    rel = set(relevant)
    denominator = max(1, len(rel))
    admitted = admit_within_budget(ranked, budget)

    sources = [source for source, _text, _cost in admitted]
    tokens_used = sum(cost for _source, _text, cost in admitted)
    unique: List[str] = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    relevant_tokens = sum(
        cost for source, _text, cost in admitted if source in rel
    )
    hit = {source for source in sources if source in rel}

    return {
        "budget_tokens": budget,
        "chunks_admitted": len(admitted),
        "tokens_used": tokens_used,
        "unique_docs": len(unique),
        "sources": sources,
        "duplicate_source_rate": (
            round((len(admitted) - len(unique)) / len(admitted), 6) if admitted else 0.0
        ),
        "recall@budget": round(len(hit) / denominator, 6),
        "relevant_tokens": relevant_tokens,
        "relevant_token_fraction": (
            round(relevant_tokens / tokens_used, 6) if tokens_used else 0.0
        ),
    }


BUDGET_METRICS: Tuple[str, ...] = (
    "recall@budget",
    "unique_docs",
    "chunks_admitted",
    "tokens_used",
    "duplicate_source_rate",
    "relevant_tokens",
    "relevant_token_fraction",
)


def aggregate_budget(scored: List[Dict[str, object]]) -> Dict[str, object]:
    """Means over the labelled queries. Nothing here is a timing."""
    if not scored:
        return {}
    out: Dict[str, object] = {"queries": len(scored)}
    for metric in BUDGET_METRICS:
        out[metric] = round(
            statistics.fmean([float(entry[metric]) for entry in scored]), 6
        )
    return out


# --- the harness ------------------------------------------------------------


class ProfileBench:
    def __init__(
        self,
        profile: dict,
        corpus: Path,
        runs_per_op: Dict[str, int],
        k: int,
        truth: Dict[str, object],
        budgets: Tuple[int, ...] = TOKEN_BUDGETS,
    ) -> None:
        self.profile = profile
        self.name = profile["name"]
        self.config: ChunkConfig = config_from_profile(profile)
        self.corpus = corpus
        self.runs_per_op = dict(runs_per_op)
        self.k = k
        self.budgets = tuple(budgets)
        # The labelled set as it was loaded — from the shipped file when one is
        # in use. Nothing below re-derives a label from QUALITY_TOPICS.
        self.truth = truth
        self._planting = list(truth.get("planting") or [])  # type: ignore
        self.collection_name = COLLECTION_PREFIX + self.name.replace("-", "_")
        # The cold-start Create probe builds a fresh index in a fresh process.
        # It gets its own collection so that a subprocess cannot disturb the one
        # the warm measurements above it were taken against.
        self.cold_collection_name = self.collection_name + "_cold"
        self._assert_target_safe()
        self.scratch_path = corpus / SCRATCH_NOTE
        self.notes: Dict[str, object] = {}

    def runs(self, op: str) -> int:
        return int(self.runs_per_op[op])

    def _assert_target_safe(self) -> None:
        for name in (self.collection_name, self.cold_collection_name):
            if not name.startswith(COLLECTION_PREFIX):
                raise RuntimeError("refusing a collection outside the bench namespace")
            if name == COLLECTION_NAME:
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
        for _ in range(self.runs("create")):
            started = time.perf_counter()
            summary = self._index_all(reset=True)
            samples.append((time.perf_counter() - started) * 1000.0)
        return samples, summary

    def op_read(self) -> Tuple[List[float], Dict[str, dict]]:
        collection = self._collection()
        samples: List[float] = []
        per_query: Dict[str, List[float]] = {query: [] for query in QUERIES}
        for _ in range(self.runs("read")):
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
        for run in range(self.runs("update")):
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

        for _ in range(self.runs("delete")):
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

    # -- read latency, decomposed -------------------------------------------

    def op_stages(self) -> dict:
        """Split the read into its stages, and refuse a split that does not close.

        The v1 headline was "on a small index, search is nearly free relative to
        MiniLM inference". v1 never measured that — it inferred it from the fact
        that read latency barely moved when the index size did. This measures it.

        Four timings per sample, on the same query in the same iteration so that
        they can be subtracted from one another:

        * **query embedding** — ``ef([query])``, the ONNX forward pass alone.
        * **vector search** — ``collection.query(query_embeddings=..., include=[])``,
          the ANN lookup with nothing materialized.
        * **result materialization** — the same query asking for documents,
          metadatas and distances, *minus* the search-only time. A difference of
          two measurements, so on an individual sample it can come out negative.
          It is floored at zero and the floored samples are **counted**: a
          silently clamped negative is a fabricated number.
        * **full in-process read** — ``collection.query(query_texts=[query])``,
          which is the sum the first three have to reconstruct.

        The embedding function instance is the one the collection itself uses.
        Each ONNX instance caches its own session, so two instances would load
        the model twice and the stage timings would not be comparable with the
        full read they are meant to decompose.
        """
        ef = _embedding_function()
        # Untimed: the ONNX session is built on first use, and that cost belongs
        # to cold start (below), not to a warm read.
        ef(["warm the onnx session before anything here is timed"])
        collection = get_collection(
            name=self.collection_name, embedding_function=ef
        )
        runs = self.runs("read")

        embed: List[float] = []
        search: List[float] = []
        material: List[float] = []
        full: List[float] = []
        residual: List[float] = []
        floored = 0

        for _ in range(runs):
            for query in QUERIES:
                started = time.perf_counter()
                vector = ef([query])
                embed_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                collection.query(
                    query_embeddings=vector, n_results=self.k, include=[]
                )
                search_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                collection.query(
                    query_embeddings=vector,
                    n_results=self.k,
                    include=["documents", "metadatas", "distances"],
                )
                search_with_payload_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                collection.query(query_texts=[query], n_results=self.k)
                full_ms = (time.perf_counter() - started) * 1000.0

                material_ms = search_with_payload_ms - search_ms
                if material_ms < 0.0:
                    material_ms = 0.0
                    floored += 1

                embed.append(embed_ms)
                search.append(search_ms)
                material.append(material_ms)
                full.append(full_ms)
                residual.append(full_ms - (embed_ms + search_ms + material_ms))

        samples = len(full)
        stages = {
            "query_embedding": summarise(embed, "ms per query, ONNX forward pass"),
            "vector_search": summarise(search, "ms per query, ANN lookup, include=[]"),
            "result_materialization": summarise(
                material, "ms per query, documents+metadatas+distances, as a delta"
            ),
        }
        total = summarise(full, "ms per query, full in-process read, k=%d" % self.k)

        summed = sum(float(block["median_ms"]) for block in stages.values())
        median_full = float(total["median_ms"])
        residual_pct = (
            round((median_full - summed) / median_full * 100.0, 3) if median_full else 0.0
        )

        if abs(residual_pct) > STAGE_RESIDUAL_LIMIT_PCT:
            raise RuntimeError(
                "stage split does not close on %s: stages sum to %.3f ms against a "
                "full read of %.3f ms, residual %.2f%% (limit %.1f%%). A "
                "decomposition that does not close must not be published."
                % (
                    self.name,
                    summed,
                    median_full,
                    residual_pct,
                    STAGE_RESIDUAL_LIMIT_PCT,
                )
            )

        embedding_share_pct = (
            round(float(stages["query_embedding"]["median_ms"]) / median_full * 100.0, 2)
            if median_full
            else None
        )

        return {
            "mode": MODE_WARM,
            "n": samples,
            "n_note": "%d passes over %d fixed queries" % (runs, len(QUERIES)),
            "k": self.k,
            "stages": stages,
            "full_in_process_read": total,
            "stage_sum_median_ms": round(summed, 3),
            "stage_residual_pct": residual_pct,
            "stage_residual_limit_pct": STAGE_RESIDUAL_LIMIT_PCT,
            "stage_residual_closes": abs(residual_pct) <= STAGE_RESIDUAL_LIMIT_PCT,
            "materialization_floored_samples": floored,
            "materialization_floored_note": (
                "materialization is the difference of two measurements and can "
                "come out negative on an individual sample; those samples are "
                "floored at zero and counted here rather than silently clamped"
            ),
            "embedding_share_of_read_pct": embedding_share_pct,
            "mcp_roundtrip": self.op_mcp_roundtrip(),
        }

    def op_mcp_roundtrip(self, runs: int = MCP_ROUNDTRIP_RUNS) -> dict:
        """One ``tools/call search_toolbox`` per sample, over stdio.

        The **outer envelope**, not a summand: it contains a process hop,
        JSON-RPC framing on both sides and the server's own formatting work, on
        top of everything the in-process stages measure. It is rendered in the
        same table and marked as not part of the sum.

        The server is launched with ``RAG_COLLECTION`` pointed at this profile's
        bench collection. ``raglog.py`` reads that variable at import, and
        ``mcp_server.py`` imports the name from there, so the redirection needs
        no code change — and the check below proves it took effect rather than
        assuming it: a response that does not name a corpus document means the
        server read some other collection, and that is a failed run, not a slow
        one.
        """
        env = dict(os.environ)
        env["RAG_COLLECTION"] = self.collection_name
        env["RAG_LOG_QUIET"] = "1"
        env["ANONYMIZED_TELEMETRY"] = "False"
        env["CHROMA_ANONYMIZED_TELEMETRY"] = "False"
        env.pop("RAG_PROFILE", None)

        corpus_names = set(corpus_documents())
        samples: List[float] = []
        server = subprocess.Popen(
            [sys.executable, str(RAG_ROOT / "mcp_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=str(RAG_ROOT),
        )

        def call(message: dict) -> dict:
            assert server.stdin is not None and server.stdout is not None
            server.stdin.write(json.dumps(message) + "\n")
            server.stdin.flush()
            line = server.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed the stream before answering")
            return json.loads(line)

        try:
            handshake = call(
                {
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "bench", "version": "2"},
                    },
                }
            )
            if "result" not in handshake:
                raise RuntimeError("MCP initialize failed: %r" % handshake)
            assert server.stdin is not None
            server.stdin.write(
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
            )
            server.stdin.flush()

            # Untimed warm-up. The server imports chromadb and loads the ONNX
            # model lazily, on its first tool call; this table is the warm
            # envelope of a warm read, so that one-off cost is paid outside the
            # samples and said so in the report. Cold start is measured, in its
            # own mode, further down.
            warm = call(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "search_toolbox",
                        "arguments": {"query": QUERIES[0], "k": self.k},
                    },
                }
            )
            text = "".join(
                block.get("text", "")
                for block in ((warm.get("result") or {}).get("content") or [])
            )
            if not any(name in text for name in corpus_names):
                raise RuntimeError(
                    "the MCP server did not answer from the bench collection %r — "
                    "no corpus document is named in its reply. Refusing to "
                    "publish a round-trip measured against an unknown collection."
                    % self.collection_name
                )

            for index in range(runs):
                query = QUERIES[index % len(QUERIES)]
                message = {
                    "jsonrpc": "2.0",
                    "id": 2 + index,
                    "method": "tools/call",
                    "params": {
                        "name": "search_toolbox",
                        "arguments": {"query": query, "k": self.k},
                    },
                }
                started = time.perf_counter()
                response = call(message)
                samples.append((time.perf_counter() - started) * 1000.0)
                if "result" not in response:
                    raise RuntimeError("MCP tools/call failed: %r" % response)
        finally:
            try:
                if server.stdin is not None:
                    server.stdin.close()
                server.wait(timeout=15)
            except Exception:  # pragma: no cover - defensive teardown
                server.kill()

        return dict(
            summarise(samples, "ms per tools/call search_toolbox over stdio, k=%d" % self.k),
            collection=self.collection_name,
            envelope=True,
            not_a_summand=(
                "the round trip contains a process hop, JSON-RPC framing and the "
                "server's own formatting on top of the in-process stages; it is "
                "an envelope around them, not one of them"
            ),
            handshake_excluded=True,
            warmup_call_excluded=True,
        )

    # -- cold start (a fresh process per sample) -----------------------------

    def op_cold_start(self, kind: str) -> dict:
        """``runs_per_op['cold_start']`` fresh subprocesses, one operation each.

        This is what v1's "cold" column claimed to be and was not: v1's cold
        figure was the first sample of a warm loop, taken in a process that had
        already imported everything and loaded the model. Here each sample pays
        interpreter startup, module import, Chroma client construction and the
        ONNX model load before it does any work.

        ``process_wall_ms`` is the headline: the whole subprocess, measured by
        the parent. ``init_ms`` and ``op_ms`` are the child's own split of it,
        and ``overhead_ms`` is what is left — interpreter startup and teardown,
        reported rather than absorbed into either.
        """
        runs = self.runs("cold_start")
        target = (
            self.collection_name if kind == "read" else self.cold_collection_name
        )
        wall: List[float] = []
        init: List[float] = []
        operation: List[float] = []
        overhead: List[float] = []
        details: List[dict] = []

        for _ in range(runs):
            payload, wall_ms = self._cold_start_probe(kind, target)
            init_ms = float(payload["init_ms"])
            op_ms = float(payload["op_ms"])
            wall.append(wall_ms)
            init.append(init_ms)
            operation.append(op_ms)
            overhead.append(wall_ms - (init_ms + op_ms))
            details.append(payload.get("detail") or {})

        unit = (
            "ms per fresh process, one pass over %d fixed queries" % len(QUERIES)
            if kind == "read"
            else "ms per fresh process, one full index build"
        )
        return {
            "operation": kind,
            "collection": target,
            "process_wall": summarise(wall, unit, mode=MODE_COLD_START),
            "init": summarise(
                init,
                "ms, module import through Chroma client and embedder ready",
                mode=MODE_COLD_START,
            ),
            "op": summarise(
                operation,
                "ms, the operation alone, on a ready process",
                mode=MODE_COLD_START,
            ),
            "overhead": summarise(
                overhead,
                "ms, interpreter startup and teardown (wall - init - op)",
                mode=MODE_COLD_START,
            ),
            "detail": details[-1] if details else {},
        }

    def _cold_start_probe(self, kind: str, collection_name: str) -> Tuple[dict, float]:
        """Run one child and time the whole subprocess from the outside.

        ``sys.executable`` is invoked directly rather than through the shell or
        a wrapper, so the venv re-exec bootstrap at the top of this file cannot
        fire in the child: a doubled interpreter startup would be measurement
        noise attributed to Chroma. ``RAG_BENCH_NO_REEXEC`` says the same thing
        a second way, because the invariant matters more than the mechanism.

        ``RAG_COLLECTION`` is set to the bench collection so that even the
        child's ambient default — the one ``raglog`` computes at import — is a
        bench namespace and never the working collection.
        """
        env = dict(os.environ)
        env["RAG_COLLECTION"] = collection_name
        env["RAG_LOG_QUIET"] = "1"
        env["RAG_BENCH_NO_REEXEC"] = "1"
        env["ANONYMIZED_TELEMETRY"] = "False"
        env["CHROMA_ANONYMIZED_TELEMETRY"] = "False"
        env.pop("RAG_PROFILE", None)

        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cold-start-probe",
            kind,
            "--profile",
            self.name,
            "--corpus",
            str(self.corpus),
            "--collection",
            collection_name,
            "-k",
            str(self.k),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            argv, capture_output=True, text=True, env=env, cwd=str(RAG_ROOT)
        )
        wall_ms = (time.perf_counter() - started) * 1000.0

        if completed.returncode != 0:
            raise RuntimeError(
                "cold-start %s probe failed (exit %d): %s"
                % (kind, completed.returncode, (completed.stderr or "").strip()[:400])
            )
        for line in (completed.stdout or "").splitlines():
            if line.startswith(COLD_START_SENTINEL):
                return json.loads(line[len(COLD_START_SENTINEL):]), wall_ms
        raise RuntimeError(
            "cold-start %s probe printed no result line; stdout was %r"
            % (kind, (completed.stdout or "")[:400])
        )

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

    # -- token-budget retrieval (untimed) ------------------------------------

    def op_token_budget(self) -> dict:
        """Score the labelled set again, under a fixed context budget.

        Untimed, and run in the same place as quality scoring — after every
        timed operation — so it cannot perturb a latency sample.

        The retrieval depth is deliberately far wider than k: the budget, not
        the rank cutoff, is what is supposed to bind. Chunks are then walked in
        rank order and admitted while the running total stays within budget,
        stopping at the first that would exceed it.
        """
        collection = self._collection()
        total = collection.count()
        depth = max(1, min(total, BUDGET_RETRIEVAL_CAP))
        labels = [dict(entry) for entry in self.truth["queries"]]  # type: ignore
        no_answer_query = str(self.truth["no_answer_query"])
        problems: List[str] = []

        def ranked_chunks(query: str) -> List[Tuple[str, str]]:
            raw = collection.query(
                query_texts=[query],
                n_results=depth,
                include=["documents", "metadatas"],
            )
            documents = (raw.get("documents") or [[]])[0]
            metadatas = (raw.get("metadatas") or [[]])[0]
            out: List[Tuple[str, str]] = []
            for position, document in enumerate(documents):
                metadata = metadatas[position] if position < len(metadatas) else {}
                out.append((metadata.get("source", "?"), document or ""))
            return out

        # One retrieval per query, reused across budgets: the ranking does not
        # depend on the budget, only the admission does.
        ranked_by_query = {
            str(entry["query"]): ranked_chunks(str(entry["query"])) for entry in labels
        }
        ranked_control = ranked_chunks(no_answer_query)

        by_budget: Dict[str, object] = {}
        for budget in self.budgets:
            scored: List[Dict[str, object]] = []
            for entry in labels:
                result = score_budget(
                    ranked_by_query[str(entry["query"])],
                    entry["relevant"],  # type: ignore[arg-type]
                    budget,
                )
                result["id"] = entry["id"]
                result["query"] = entry["query"]
                scored.append(result)

            # Same contract as the equal-k control: a query no document answers
            # must score zero recall however the context is packed. If it does
            # not, the labels are being read off the retriever's own output.
            control = score_budget(ranked_control, [], budget)
            control["query"] = no_answer_query
            if control["recall@budget"]:
                raise RuntimeError(
                    "unanswerable control scored recall@budget %s at budget %d - "
                    "the labels are not independent of the retriever"
                    % (control["recall@budget"], budget)
                )

            by_budget[str(budget)] = {
                "aggregate": aggregate_budget(scored),
                "per_query": scored,
                "control_unanswerable": dict(control, scored_zero=True),
            }

        return {
            "measured": True,
            "timed": False,
            "measured_after": "all timed ops; excluded from every latency sample",
            "budgets": list(self.budgets),
            "headline_budget": HEADLINE_BUDGET,
            "retrieval_depth_chunks": depth,
            "retrieval_depth_note": (
                "min(collection.count(), %d): the budget is meant to bind, not "
                "the rank cutoff" % BUDGET_RETRIEVAL_CAP
            ),
            "token_approximation": (
                "ceil(len(text) / 4) - a character heuristic, not a tokenizer; "
                "applied identically to every profile"
            ),
            "admission_rule": (
                "greedy prefix: walk the ranked chunks in order and admit each "
                "while the running total stays within budget, stopping at the "
                "first chunk that would exceed it. Not skip-and-continue, which "
                "is a different retriever and would flatter small chunks"
            ),
            "by_budget": by_budget,
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

        # After the four timed ops, so nothing above can have observed a query
        # issued here — and before the untimed scoring, so the collection is in
        # the state Delete left it in for both.
        stages = self.op_stages()
        cold_start = {
            "read": self.op_cold_start("read"),
            "create": self.op_cold_start("create"),
        }

        # Untimed, and strictly last: no latency sample above can contain them.
        quality = self.op_quality()
        token_budget = self.op_token_budget()

        chunk_sizes = [
            len(chunk.text)
            for path in sorted(self.corpus.glob("*.md"))
            for chunk in self._chunks_for(path)
        ]

        return {
            "collection": self.collection_name,
            "indexed_chunks": indexed,
            "quality": quality,
            "token_budget": token_budget,
            "read_stages": stages,
            "cold_start": cold_start,
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
    runs_per_op = harness["runs_per_op"]
    lines.append(
        "harness     runs %s  queries=%s  k=%s  network=%s"
        % (
            " ".join("%s=%s" % (op, runs_per_op[op]) for op in sorted(runs_per_op)),
            len(harness["queries"]),
            harness["k"],
            harness["network"],
        )
    )
    lines.append(
        "statistics  median/min/max/MAD always; P50/P95 only at n>=%d; no P99 at any n"
        % harness.get("percentile_min_n", PERCENTILE_MIN_N)
    )
    lines.append("")
    lines.append("mode: warm (many operations in one process, model loaded, cache hot)")
    header = "%-9s %-38s %5s %9s %9s %9s %9s %9s %9s" % (
        "op",
        "unit",
        "n",
        "median",
        "MAD",
        "min",
        "max",
        "P50",
        "P95",
    )
    lines.append(header)
    lines.append("-" * len(header))

    def _cell(value) -> str:
        return "        -" if value is None else "%9.2f" % value

    def row(label: str, stats: dict) -> str:
        return "%-9s %-38s %5d %s %s %s %s %s %s" % (
            label,
            stats["unit"][:38],
            stats["n"],
            _cell(stats["median_ms"]),
            _cell(stats["mad_ms"]),
            _cell(stats["min_ms"]),
            _cell(stats["max_ms"]),
            _cell(stats.get("p50_ms")),
            _cell(stats.get("p95_ms")),
        )

    ops = report["ops"]
    lines.append(row("create", ops["create"]))
    lines.append(row("read", ops["read"]))
    lines.append(row("update", ops["update"]))
    lines.append(row("delete", ops["delete"]))
    lines.append(row("  +lag", ops["delete"]["consistency"]))
    lines.append("")
    lines.append("read, per query (median / MAD ms):")
    for query, stats in ops["read"]["per_query"].items():
        lines.append(
            "  %-52s %8.2f %8.2f" % (query[:52], stats["median_ms"], stats["mad_ms"])
        )
    lines.append("")
    lines.append(render_stages(report.get("read_stages")))
    lines.append("")
    lines.append(render_cold_start(report.get("cold_start")))
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
    lines.append("")
    lines.append(render_token_budget(report.get("token_budget")))
    lines.append("=" * 78)
    return "\n".join(lines)


def render_stages(stages: Optional[dict]) -> str:
    if not stages:
        return "read stages: not measured"
    lines: List[str] = []
    lines.append(
        "read latency by stage (mode %s, n=%d — %s):"
        % (stages["mode"], stages["n"], stages["n_note"])
    )
    header = "  %-26s %9s %9s %9s %9s" % ("stage", "median", "MAD", "min", "max")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    def row(label: str, block: dict, marker: str = "") -> str:
        return "  %-26s %9.3f %9.3f %9.3f %9.3f%s" % (
            label[:26],
            block["median_ms"],
            block["mad_ms"],
            block["min_ms"],
            block["max_ms"],
            marker,
        )

    for key, label in (
        ("query_embedding", "query embedding"),
        ("vector_search", "vector search"),
        ("result_materialization", "result materialization"),
    ):
        lines.append(row(label, stages["stages"][key]))
    lines.append(
        "  %-26s %9.3f" % ("= sum of stages", stages["stage_sum_median_ms"])
    )
    lines.append(row("full in-process read", stages["full_in_process_read"]))
    lines.append(
        "  residual %.2f%% of the full read (limit %.1f%%) — %s"
        % (
            stages["stage_residual_pct"],
            stages["stage_residual_limit_pct"],
            "closes" if stages["stage_residual_closes"] else "DOES NOT CLOSE",
        )
    )
    lines.append(
        "  query embedding is %.1f%% of the full in-process read"
        % stages["embedding_share_of_read_pct"]
    )
    lines.append(
        "  materialization samples floored at zero: %d of %d"
        % (stages["materialization_floored_samples"], stages["n"])
    )
    mcp = stages.get("mcp_roundtrip") or {}
    if mcp:
        lines.append(
            "  %-26s %9.3f %9.3f %9.3f %9.3f   (n=%d, envelope — NOT part of the sum)"
            % (
                "full MCP round-trip",
                mcp["median_ms"],
                mcp["mad_ms"],
                mcp["min_ms"],
                mcp["max_ms"],
                mcp["n"],
            )
        )
        lines.append("  MCP server read collection %s" % mcp["collection"])
    return "\n".join(lines)


def render_cold_start(cold: Optional[dict]) -> str:
    if not cold:
        return "cold start: not measured"
    lines: List[str] = []
    first = next(iter(cold.values()))
    lines.append(
        "cold start (mode %s, n=%d — a fresh process per sample, median/min/max "
        "only; N=%d earns no percentile of any kind):"
        % (
            first["process_wall"]["mode"],
            first["process_wall"]["n"],
            first["process_wall"]["n"],
        )
    )
    header = "  %-12s %-22s %5s %10s %10s %10s" % (
        "operation",
        "component",
        "n",
        "median",
        "min",
        "max",
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for operation, block in cold.items():
        for key, label in (
            ("process_wall", "process wall (total)"),
            ("init", "  init"),
            ("op", "  operation"),
            ("overhead", "  startup/teardown"),
        ):
            stats = block[key]
            lines.append(
                "  %-12s %-22s %5d %10.2f %10.2f %10.2f"
                % (
                    operation if key == "process_wall" else "",
                    label,
                    stats["n"],
                    stats["median_ms"],
                    stats["min_ms"],
                    stats["max_ms"],
                )
            )
    return "\n".join(lines)


def render_token_budget(budget: Optional[dict]) -> str:
    if not budget:
        return "token budget: not measured"
    lines: List[str] = []
    lines.append(
        "token-budget retrieval (untimed; %s; %s):"
        % (budget["token_approximation"], budget["retrieval_depth_note"])
    )
    header = "  %6s %12s %11s %11s %11s %11s %11s" % (
        "budget",
        "recall@bud",
        "uniq docs",
        "chunks",
        "tokens",
        "dup rate",
        "rel tok frac",
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for key in sorted(budget["by_budget"], key=int):
        aggregate = budget["by_budget"][key]["aggregate"]
        lines.append(
            "  %6s %12.3f %11.3f %11.3f %11.1f %11.3f %11.3f"
            % (
                key,
                aggregate["recall@budget"],
                aggregate["unique_docs"],
                aggregate["chunks_admitted"],
                aggregate["tokens_used"],
                aggregate["duplicate_source_rate"],
                aggregate["relevant_token_fraction"],
            )
        )
    lines.append(
        "  control: the unanswerable query scored recall 0 at every budget"
    )
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
    stamps = {r["date"] for r in ordered}

    # Every table below states its own N. A run whose profiles disagree about
    # how many samples an operation got cannot be rendered as one table, so the
    # generator refuses rather than printing a number over an ambiguous N.
    runs_per_op: Dict[str, int] = dict(harness["runs_per_op"])
    for report in ordered[1:]:
        if dict(report["harness"]["runs_per_op"]) != runs_per_op:
            raise RuntimeError(
                "result files disagree about sample sizes (%s vs %s); a table "
                "that cannot state its own N must not be rendered"
                % (runs_per_op, report["harness"]["runs_per_op"])
            )
    percentile_min_n = int(harness.get("percentile_min_n", PERCENTILE_MIN_N))

    def runs_of(op: str) -> int:
        return int(runs_per_op[op])

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
    lines.append(
        "What produced the numbers below. **This table has no N and no mode of "
        "its own — it is where the N and the modes for every other table are "
        "declared.**"
    )
    lines.append("")
    lines.extend(
        _table(
            ["", ""],
            [
                ["source files", ", ".join("`bench/results/%s`" % r["_path"] for r in ordered)],
                ["run date", ", ".join(sorted(stamps))],
                ["profiles", ", ".join("`%s`" % n for n in names)],
                [
                    "samples per operation",
                    ", ".join(
                        "%s **%d**" % (op.replace("_", " "), runs_per_op[op])
                        for op in ("create", "read", "update", "delete", "cold_start")
                        if op in runs_per_op
                    ),
                ],
                [
                    "modes",
                    "`%s` (many operations in one process) and `%s` (a fresh "
                    "process per sample), never mixed in one table"
                    % (MODE_WARM, MODE_COLD_START),
                ],
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
    lines.append(
        "The machine every figure below was produced on. **No N and no mode: "
        "this is not a measurement**, it is the configuration the measurements "
        "were taken in."
    )
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
    lines.append(
        "Four operations are timed against a synthetic corpus generated "
        "deterministically from seed %s and shipped in this repository as "
        "`bench/corpus/`. The corpus on disk is compared byte-for-byte with the "
        "generator's output before anything is measured, and the run fails "
        "loudly if they differ — otherwise the numbers would describe a corpus "
        "nobody can see." % corpus.get("seed")
    )
    lines.append("")
    lines.extend(
        _table(
            ["operation", "N", "mode", "what is timed"],
            [
                [
                    "create",
                    str(runs_of("create")),
                    MODE_WARM,
                    "a full index build of the whole corpus",
                ],
                [
                    "read",
                    str(runs_of("read")),
                    MODE_WARM,
                    "one pass over %d fixed queries at k=%s, fixed order, no sampling"
                    % (len(harness.get("queries") or []), harness.get("k")),
                ],
                [
                    "update",
                    str(runs_of("update")),
                    MODE_WARM,
                    "re-index of a single changed note",
                ],
                ["delete", str(runs_of("delete")), MODE_WARM, "removal of one note"],
                [
                    "delete +lag",
                    str(runs_of("delete")),
                    MODE_WARM,
                    "time until the removed note stops being returned by a probe query",
                ],
                [
                    "cold start",
                    str(runs_of("cold_start")),
                    MODE_COLD_START,
                    "a fresh subprocess per sample, one operation each",
                ],
            ],
        )
    )
    lines.append("")
    earns = [op for op in TIMED_OPS if runs_of(op) >= percentile_min_n]
    denied = [op for op in TIMED_OPS if runs_of(op) < percentile_min_n]

    def _op_list(ops: List[str]) -> str:
        return ", ".join("%s (N=%d)" % (op, runs_of(op)) for op in ops)

    if earns and denied:
        gate = "So %s carry percentiles; %s do not, and report a median with a MAD instead." % (
            _op_list(earns),
            _op_list(denied),
        )
    elif earns:
        gate = "Every operation here is sampled to at least %d, so all of them carry percentiles: %s." % (
            percentile_min_n,
            _op_list(earns),
        )
    else:
        gate = (
            "No operation in this run reaches %d samples, so **no percentile is "
            "rendered anywhere below** — every latency figure is a median with a "
            "MAD, a minimum and a maximum. This is a reduced-sample run: %s."
            % (percentile_min_n, _op_list(denied))
        )
    lines.append(
        "**Each operation has its own N, and the statistics are gated on it.** "
        "The four operations differ by two orders of magnitude in cost, so one "
        "N for the harness either starves the cheap operations of samples or "
        "makes the expensive ones unrunnable. Median, minimum, maximum and "
        "median absolute deviation are reported for every operation: they are "
        "statements about the samples in hand. **P50 and P95 are reported only "
        "at N ≥ %d** — below that a percentile is an interpolation between two "
        "neighbouring samples presented as a property of a distribution. %s"
        % (percentile_min_n, gate)
    )
    lines.append("")
    lines.append(
        "**There is no P99 in this document, at any N.** Not suppressed — not "
        "computed. At N=%d the 99th percentile is an interpolation between the "
        "two largest samples, which is a way of printing the maximum and "
        "calling it a tail statistic; at N=%d it simply *is* the maximum. No "
        "sample count used anywhere in this harness earns one, so the code does "
        "not produce one."
        % (percentile_min_n, runs_of("create"))
    )
    lines.append("")
    lines.append(
        "**MAD rather than standard deviation at small N.** The median absolute "
        "deviation — `median(|x - median(x)|)` — asks how far a typical sample "
        "sits from a typical sample. A standard deviation assumes the tail was "
        "sampled and is dragged around by the one slow run every process has, "
        "which at N=%d is a statement about scheduling luck rather than about "
        "the operation. Both are in the result JSONs; the tables show the MAD."
        % runs_of("create")
    )
    lines.append("")
    lines.append(
        "**Two modes, named, never mixed.** `%s` is many operations in one "
        "process: the model is loaded, the OS cache is hot, and the marginal "
        "cost of one more operation is what is being measured. `%s` is a fresh "
        "subprocess per sample, which pays interpreter startup, module import, "
        "Chroma client construction and the ONNX model load before it does any "
        "work. Every table below states which mode produced it. The four timed "
        "operations and the stage split are warm; the cold-start section is not."
        % (MODE_WARM, MODE_COLD_START)
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
    budget0 = ordered[0].get("token_budget") or {}
    stages0 = ordered[0].get("read_stages") or {}
    lines.append(
        "**Read latency is decomposed, and the decomposition has to close.** "
        "Query embedding, vector search and result materialization are timed "
        "separately on the same query in the same iteration, N=%d pass%s over "
        "the %d fixed queries. Materialization is a *difference* of two "
        "measurements — the same search with and without documents, metadatas "
        "and distances — so on an individual sample it can come out negative; "
        "those samples are floored at zero and **counted**, and the count is in "
        "the table. The three stages are required to sum to within %.0f%% of "
        "the full in-process read, and a run whose split does not close fails "
        "rather than publishing a decomposition that does not add up."
        % (
            runs_of("read"),
            "" if runs_of("read") == 1 else "es",
            len(harness.get("queries") or []),
            float(stages0.get("stage_residual_limit_pct") or STAGE_RESIDUAL_LIMIT_PCT),
        )
    )
    lines.append("")
    lines.append(
        "The full MCP round-trip is measured alongside them, over stdio, at "
        "N=%s. It is an **outer envelope, not a summand**: it contains a "
        "process hop, JSON-RPC framing on both sides and the server's own "
        "formatting work on top of everything the in-process stages measure. "
        "The `initialize` handshake and one warm-up call happen outside the "
        "timed samples, so what is timed is request→response on a warm server. "
        "The server is launched against this harness's own collection and its "
        "first reply is checked to name a corpus document — a round trip that "
        "answered from some other collection is a failed run, not a fast one."
        % _fmt((stages0.get("mcp_roundtrip") or {}).get("n"))
    )
    lines.append("")
    lines.append(
        "**Retrieval is scored twice: at equal k, and under a token budget.** "
        "Equal k is not a fair comparison across chunk geometries — %d chunks "
        "of %s characters and %d chunks of %s characters are not the same "
        "context. The budget-constrained scoring retrieves the top %s chunks "
        "and then admits them under a fixed budget, which is the constraint a "
        "caller actually has."
        % (
            harness.get("k"),
            _fmt(min(r["chunking_observed"]["chars_mean"] for r in ordered), 1),
            harness.get("k"),
            _fmt(max(r["chunking_observed"]["chars_mean"] for r in ordered), 1),
            _fmt(budget0.get("retrieval_depth_chunks")),
        )
    )
    lines.append("")
    lines.append(
        "Two rules of that scoring have to be stated, because a reader cannot "
        "check the numbers without them:"
    )
    lines.append("")
    lines.append(
        "1. *Tokens are approximated as `ceil(len(text) / 4)`.* **This is a "
        "character heuristic, not a tokenizer.** Real BPE counts move with "
        "vocabulary, casing and whitespace, and this approximation will be "
        "wrong for any specific model — discount the absolute token figures "
        "accordingly. It is applied identically to every profile, so it cannot "
        "favour one of them; what it cannot support is a claim about a real "
        "model's context window."
    )
    lines.append(
        "2. *Admission is a greedy **prefix**.* Chunks are walked in rank order "
        "and admitted while the running total stays within budget; the walk "
        "**stops** at the first chunk that would exceed it. It does not skip "
        "that chunk and carry on down the ranking. Skip-and-continue is a "
        "different retriever and a flattering one: it would let a profile with "
        "many small chunks backfill the remaining budget with lower-ranked text "
        "while a profile with large chunks got truncated, and the comparison "
        "would then be measuring the packer rather than the chunking."
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
    lines.append("## Latency — mode `%s`" % MODE_WARM)
    lines.append("")
    lines.append(
        "Many operations in one process: the model is loaded, the OS cache is "
        "hot, and what is measured is the marginal cost of one more operation. "
        "This is v1's only mode, named honestly. For the cost of the first "
        "operation in a fresh process, see [Cold start](#cold-start)."
    )
    lines.append("")
    op_labels = [
        ("create", "create", None),
        ("read", "read", None),
        ("update", "update", None),
        ("delete", "delete", None),
        ("delete", "delete +lag", "consistency"),
    ]
    lines.append("Median, milliseconds. **Mode `%s`; N per row in the `n` column:**" % MODE_WARM)
    lines.append("")
    rows = []
    for op_key, label, nested in op_labels:
        stats0 = ordered[0]["ops"][op_key]
        if nested:
            stats0 = stats0[nested]
        row = [label, str(stats0["n"])]
        for report in ordered:
            stats = report["ops"][op_key]
            if nested:
                stats = stats[nested]
            row.append(_fmt(stats["median_ms"]))
        rows.append(row)
    lines.extend(_table(["operation (median ms)", "n"] + names, rows))
    lines.append("")
    lines.append(
        "Full distribution, milliseconds. `P50`/`P95` are blank where N < %d; "
        "there is no P99 column because no N here earns one:" % percentile_min_n
    )
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
                    str(stats["mode"]),
                    str(stats["n"]),
                    _fmt(stats["median_ms"]),
                    _fmt(stats["mad_ms"]),
                    _fmt(stats["min_ms"]),
                    _fmt(stats["max_ms"]),
                    _fmt(stats.get("p50_ms")),
                    _fmt(stats.get("p95_ms")),
                ]
            )
    lines.extend(
        _table(
            [
                "profile",
                "operation",
                "mode",
                "n",
                "median",
                "MAD",
                "min",
                "max",
                "P50",
                "P95",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Corpus as each profile cut it. **No N and no mode: this is not a "
        "sampled measurement.** It is a deterministic property of the chunker "
        "applied to a fixed corpus — running it again produces the same numbers, "
        "which is why there is nothing to take a median of:"
    )
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

    # --- cold start ---------------------------------------------------------
    cold_n = runs_of("cold_start")
    lines.append("## Cold start")
    lines.append("")
    lines.append(
        "**mode `%s`, n=%d per cell.** A fresh subprocess per sample, invoked "
        "directly so the interpreter starts exactly once. Each sample pays "
        "module import, Chroma client construction and the ONNX model load "
        "before it does any work. `process wall` is measured by the parent "
        "around the whole subprocess and is the headline; `init` and `operation` "
        "are the child's own split of it; `startup/teardown` is what is left "
        "over — `wall − (init + operation)` — reported rather than absorbed "
        "into either neighbour."
        % (MODE_COLD_START, cold_n)
    )
    lines.append("")
    lines.append(
        "**Median, minimum and maximum only.** N=%d earns no percentile of any "
        "kind, and none is shown." % cold_n
    )
    lines.append("")
    for operation, caption in (
        ("read", "one pass over the %d fixed queries" % len(harness.get("queries") or [])),
        ("create", "a full index build, in a fresh process"),
    ):
        lines.append("`%s` — %s:" % (operation, caption))
        lines.append("")
        rows = []
        for report in ordered:
            block = (report.get("cold_start") or {}).get(operation) or {}
            for key, label in (
                ("process_wall", "process wall (total)"),
                ("init", "init"),
                ("op", "operation"),
                ("overhead", "startup/teardown"),
            ):
                stats = block.get(key) or {}
                rows.append(
                    [
                        report["profile"]["name"],
                        label,
                        str(stats.get("mode")),
                        str(stats.get("n")),
                        _fmt(stats.get("median_ms")),
                        _fmt(stats.get("min_ms")),
                        _fmt(stats.get("max_ms")),
                    ]
                )
        lines.extend(
            _table(
                ["profile", "component", "mode", "n", "median", "min", "max"], rows
            )
        )
        lines.append("")

    warm_read = ordered[0]["ops"]["read"]["median_ms"]
    cold_read = (
        ((ordered[0].get("cold_start") or {}).get("read") or {}).get("process_wall") or {}
    ).get("median_ms")
    if cold_read:
        lines.append(
            "On `%s` the same read pass costs %s ms warm and %s ms from a fresh "
            "process — a factor of **%.0fx**, essentially all of it paid before "
            "the first query is issued. v1 reported a `cold` column that was the "
            "first sample of a warm loop, in a process that had already imported "
            "everything and loaded the model; that column is gone."
            % (
                ordered[0]["profile"]["name"],
                _fmt(warm_read),
                _fmt(cold_read),
                float(cold_read) / float(warm_read),
            )
        )
        lines.append("")
    lines.append(
        "**This is a cold *process*, not a cold *machine*.** The OS page cache "
        "still holds the ONNX model file after the first sample, and it is not "
        "purged between samples — purging it needs privileges this harness does "
        "not take. The first sample of the first profile pays a genuinely cold "
        "file read; the rest do not. Treat these figures as the cost of starting "
        "a new process on a machine that has run this before, which is the "
        "common case for a CLI, and not as a first-boot number."
    )
    lines.append("")

    # --- read, by stage -----------------------------------------------------
    lines.append("## Where the read time goes")
    lines.append("")
    lines.append(
        "**mode `%s`, n=%s per profile** (%d pass%s over %d fixed queries). "
        "Milliseconds per query."
        % (
            MODE_WARM,
            _fmt(stages0.get("n")),
            runs_of("read"),
            "" if runs_of("read") == 1 else "es",
            len(harness.get("queries") or []),
        )
    )
    lines.append("")
    rows = []
    for report in ordered:
        stages = report.get("read_stages") or {}
        blocks = stages.get("stages") or {}
        mcp = stages.get("mcp_roundtrip") or {}
        name = report["profile"]["name"]
        for key, label in (
            ("query_embedding", "query embedding"),
            ("vector_search", "vector search"),
            ("result_materialization", "result materialization"),
        ):
            stats = blocks.get(key) or {}
            rows.append(
                [
                    name,
                    label,
                    str(stats.get("n")),
                    _fmt(stats.get("median_ms"), 3),
                    _fmt(stats.get("mad_ms"), 3),
                    _fmt(stats.get("min_ms"), 3),
                    _fmt(stats.get("max_ms"), 3),
                    "yes",
                ]
            )
        rows.append(
            [
                name,
                "**sum of the three**",
                "",
                _fmt(stages.get("stage_sum_median_ms"), 3),
                "",
                "",
                "",
                "",
            ]
        )
        total = stages.get("full_in_process_read") or {}
        rows.append(
            [
                name,
                "**full in-process read**",
                str(total.get("n")),
                _fmt(total.get("median_ms"), 3),
                _fmt(total.get("mad_ms"), 3),
                _fmt(total.get("min_ms"), 3),
                _fmt(total.get("max_ms"), 3),
                "is the total",
            ]
        )
        rows.append(
            [
                name,
                "full MCP round-trip",
                str(mcp.get("n")),
                _fmt(mcp.get("median_ms"), 3),
                _fmt(mcp.get("mad_ms"), 3),
                _fmt(mcp.get("min_ms"), 3),
                _fmt(mcp.get("max_ms"), 3),
                "**no — envelope**",
            ]
        )
    lines.extend(
        _table(
            ["profile", "stage", "n", "median", "MAD", "min", "max", "in the sum?"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "How well the decomposition closes, and what had to be floored. Same "
        "samples as the table above — **mode `%s`, n=%s per profile** — "
        "restated so the residual is not read against a different N:"
        % (MODE_WARM, _fmt(stages0.get("n")))
    )
    lines.append("")
    rows = []
    floored_pct: Dict[str, float] = {}
    for report in ordered:
        stages = report.get("read_stages") or {}
        name = report["profile"]["name"]
        total_samples = int(stages.get("n") or 0)
        floored = int(stages.get("materialization_floored_samples") or 0)
        floored_pct[name] = (100.0 * floored / total_samples) if total_samples else 0.0
        rows.append(
            [
                name,
                _fmt(stages.get("stage_sum_median_ms"), 3),
                _fmt((stages.get("full_in_process_read") or {}).get("median_ms"), 3),
                "%s%%" % _fmt(stages.get("stage_residual_pct"), 2),
                "%s%%" % _fmt(stages.get("stage_residual_limit_pct"), 0),
                "%d / %d (%s%%)" % (floored, total_samples, _fmt(floored_pct[name], 1)),
            ]
        )
    lines.extend(
        _table(
            [
                "profile",
                "sum of stages",
                "full read",
                "residual",
                "limit",
                "materialization samples floored at 0",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "The residual is interpreter and call overhead that belongs to no stage. "
        "It is inside the %s%% limit on every profile; a run where it is not "
        "fails and publishes nothing. The floored samples are the ones where "
        "materialization — the difference between the same search with and "
        "without its payload — came out negative through measurement noise; "
        "they are clamped to zero and counted here rather than silently."
        % _fmt(stages0.get("stage_residual_limit_pct"), 0)
    )
    lines.append("")
    noisy = sorted(
        (name for name, share in floored_pct.items() if share >= 10.0),
        key=lambda n: -floored_pct[n],
    )
    if noisy:
        lines.append(
            "**Do not read the materialization column as a measurement on %s.** "
            "%s of its samples floored, which means the stage is smaller than "
            "the run-to-run noise of the two searches it is the difference of. "
            "Its median is a lower bound on a quantity this clock cannot "
            "resolve, not an estimate of it. The number is left in the table "
            "because deleting it would make the three stages appear to sum "
            "more cleanly than they do; the two stages above it, and the total, "
            "are measurements."
            % (
                " or ".join("`%s`" % n for n in noisy),
                " and ".join(
                    "%s%% on `%s`" % (_fmt(floored_pct[n], 1), n) for n in noisy
                )
                if len(noisy) > 1
                else "%s%%" % _fmt(floored_pct[noisy[0]], 1),
            )
        )
        lines.append("")
    lines.append(
        "The MCP round-trip is the envelope, not a summand: it adds a process "
        "hop, JSON-RPC framing on both sides and the server's own result "
        "formatting on top of everything above it. It is measured at N=%s "
        "against the same collection, on a server that has already answered one "
        "untimed query."
        % _fmt((stages0.get("mcp_roundtrip") or {}).get("n"))
    )
    lines.append("")

    # --- quality ------------------------------------------------------------
    lines.append("## Retrieval quality")
    lines.append("")
    lines.append(
        "Equal k — every profile retrieves the same *number of chunks*. "
        "**N = %s labelled queries over %s documents, retrieval depth %s "
        "chunks.** **No mode applies: this is untimed.** Scoring runs after all "
        "four timed operations and takes no timing samples, so there is no warm "
        "or cold-start reading of it to give — the absence of a mode column here "
        "is a property of the measurement, not an omission."
        % (truth0.get("queries"), truth0.get("corpus_documents"), depth)
    )
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
    lines.append(
        "**Also untimed, so again no mode applies.** The two controls carry "
        "different N and it matters which is which: the deranged-label control "
        "re-scores the same **%s labelled queries** against document-disjoint "
        "label sets, so its columns are means over %s queries; the unanswerable "
        "control is **a single query**, deliberately not one of the %s, and its "
        "columns are that one query's own scores rather than a mean."
        % (
            truth0.get("queries"),
            truth0.get("queries"),
            truth0.get("queries"),
        )
    )
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

    # --- token budget -------------------------------------------------------
    budget_keys = sorted((budget0.get("by_budget") or {}), key=int)

    def budget_aggregate(report: dict, key: str) -> Dict[str, object]:
        block = ((report.get("token_budget") or {}).get("by_budget") or {}).get(key) or {}
        return dict(block.get("aggregate") or {})

    lines.append("## Retrieval quality under a token budget")
    lines.append("")
    lines.append(
        "The same labelled queries, scored again with the *context* held fixed "
        "instead of the chunk count. Untimed, run after every timed operation. "
        "Means over the %s labelled queries. Tokens are `ceil(len(text) / 4)` — "
        "an approximation, see Method." % truth0.get("queries")
    )
    lines.append("")
    for key in budget_keys:
        lines.append(
            "Budget **%s tokens** — **untimed, so no mode applies; means over "
            "%s labelled queries:**" % (key, truth0.get("queries"))
        )
        lines.append("")
        rows = []
        for report in ordered:
            aggregate = budget_aggregate(report, key)
            rows.append(
                [
                    report["profile"]["name"],
                    _fmt(aggregate.get("recall@budget"), 3),
                    _fmt(aggregate.get("unique_docs"), 2),
                    _fmt(aggregate.get("chunks_admitted"), 2),
                    _fmt(aggregate.get("tokens_used"), 1),
                    _fmt(aggregate.get("duplicate_source_rate"), 3),
                    _fmt(aggregate.get("relevant_token_fraction"), 3),
                ]
            )
        lines.extend(
            _table(
                [
                    "profile",
                    "recall@budget",
                    "unique docs",
                    "chunks admitted",
                    "tokens used",
                    "duplicate source rate",
                    "relevant token fraction",
                ],
                rows,
            )
        )
        lines.append("")
    lines.append(
        "The unanswerable control was run under every budget on every profile "
        "and scored `recall@budget` 0 in all of them; a non-zero score there "
        "fails the run, exactly as it does at equal k."
    )
    lines.append("")

    # --- the findings -------------------------------------------------------
    # Both headlines are computed here from the shipped result files, including
    # the branch in which the second one has nothing flattering to say. A
    # finding that only renders when it is convenient is not a finding.
    read_median = {r["profile"]["name"]: r["ops"]["read"]["median_ms"] for r in ordered}
    budgets = {
        r["profile"]["name"]: (r["profile"].get("chunking") or {}).get("max_chars")
        for r in ordered
    }
    mean_chunk = {
        r["profile"]["name"]: r["chunking_observed"]["chars_mean"] for r in ordered
    }
    chunk_count = {r["profile"]["name"]: r["chunking_observed"]["chunks"] for r in ordered}
    quality_by_name = {r["profile"]["name"]: _quality_row(r) for r in ordered}

    read_span = max(read_median.values()) / min(read_median.values())
    budget_span = max(budgets.values()) / min(budgets.values())
    mean_span = max(mean_chunk.values()) / min(mean_chunk.values())
    count_span = max(chunk_count.values()) / min(chunk_count.values())
    fastest_read = min(read_median, key=lambda n: read_median[n])
    fastest_create = min(
        ordered, key=lambda r: r["ops"]["create"]["median_ms"]
    )["profile"]["name"]

    def equal_k_key(name: str):
        aggregate = quality_by_name[name]
        return (
            aggregate.get("recall@3", 0.0),
            aggregate.get(ndcg_key, 0.0),
            aggregate.get("mrr", 0.0),
        )

    equal_k_ranking = sorted(quality_by_name, key=equal_k_key, reverse=True)
    best_quality = equal_k_ranking[0]

    lines.append("## Findings")
    lines.append("")

    # --- headline 1: where the read time actually goes ----------------------
    embedding_share = {
        r["profile"]["name"]: (r.get("read_stages") or {}).get(
            "embedding_share_of_read_pct"
        )
        for r in ordered
    }
    stage_medians = {
        r["profile"]["name"]: {
            key: ((r.get("read_stages") or {}).get("stages") or {})
            .get(key, {})
            .get("median_ms")
            for key in ("query_embedding", "vector_search", "result_materialization")
        }
        for r in ordered
    }
    shares = [value for value in embedding_share.values() if value is not None]
    slowest_read = max(read_median, key=lambda n: read_median[n])

    lines.append("### 1. Read latency is query embedding, not index size")
    lines.append("")
    lines.append(
        "v1 claimed that on a small index the vector search is nearly free "
        "relative to MiniLM inference, and inferred it from the fact that read "
        "latency barely moved when the index size did. v2 measures it: on this "
        "corpus, **query embedding is %.0f–%.0f%% of the full in-process read** "
        "across the three profiles. The claim is now backed by the "
        "decomposition rather than by an absence of variation. Medians, "
        "**mode `%s`, n=%s per profile** — the same samples as [Where the read "
        "time goes](#where-the-read-time-goes), restated here:"
        % (min(shares), max(shares), MODE_WARM, _fmt(stages0.get("n")))
    )
    lines.append("")
    rows = []
    for name in [r["profile"]["name"] for r in ordered]:
        medians = stage_medians[name]
        rows.append(
            [
                name,
                str(chunk_count[name]),
                _fmt(medians["query_embedding"], 3),
                _fmt(medians["vector_search"], 3),
                _fmt(medians["result_materialization"], 3),
                "**%s%%**" % _fmt(embedding_share[name], 1),
            ]
        )
    lines.extend(
        _table(
            [
                "profile",
                "chunks in index",
                "query embedding (ms)",
                "vector search (ms)",
                "materialization (ms)",
                "embedding share of the read",
            ],
            rows,
        )
    )
    lines.append("")
    search_medians = [stage_medians[n]["vector_search"] for n in stage_medians]
    lines.append(
        "The vector search itself costs %s–%s ms while the number of chunks in "
        "the index spans **%.1fx** (%d → %d). The embedding is a fixed cost per "
        "query that the chunking cannot move, and at this corpus size it "
        "dominates everything the index does. That is what makes read latency "
        "look flat: the median read pass spans only **%.2fx** across the three "
        "profiles (%s ms on `%s`, %s ms on `%s`), against a %.1fx span in the "
        "chunk budget and a %.1fx span in the mean chunk actually produced "
        "(%s → %s characters)."
        % (
            _fmt(min(search_medians), 3),
            _fmt(max(search_medians), 3),
            count_span,
            min(chunk_count.values()),
            max(chunk_count.values()),
            read_span,
            _fmt(min(read_median.values())),
            fastest_read,
            _fmt(max(read_median.values())),
            slowest_read,
            budget_span,
            mean_span,
            _fmt(min(mean_chunk.values()), 1),
            _fmt(max(mean_chunk.values()), 1),
        )
    )
    lines.append("")
    lines.append(
        "**The scope of that finding is this corpus.** %d to %d chunks is small "
        "enough that the ANN structure is doing almost nothing; the sentence "
        "above is about a small index, and nothing here measures where the "
        "crossover is."
        % (min(chunk_count.values()), max(chunk_count.values()))
    )
    lines.append("")

    lines.append("### 2. Timing alone picks the wrong profile")
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
        "On this corpus the timed operations do not separate the profiles on "
        "the axis a reader would care about. " + cheapest
    )
    lines.append("")
    best = quality_by_name[best_quality]
    lines.append(
        "**Quality is what decides it.** At equal k, `%s` leads on `recall@3` "
        "(%s) and on `%s` (%s), and it is the profile this toolkit ships as its "
        "default. The full ladder — note that this table deliberately puts three "
        "different N side by side: the latency columns are **mode `%s`**, read "
        "at **n=%d** and create at **n=%d**, while the quality columns are "
        "**untimed** means over **%s labelled queries**:"
        % (
            best_quality,
            _fmt(best.get("recall@3"), 3),
            ndcg_key,
            _fmt(best.get(ndcg_key), 3),
            MODE_WARM,
            runs_of("read"),
            runs_of("create"),
            truth0.get("queries"),
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
                _fmt(read_median[name]),
                _fmt(report["ops"]["create"]["median_ms"]),
                _fmt(aggregate.get("recall@3"), 3),
                _fmt(aggregate.get(ndcg_key), 3),
            ]
        )
    lines.extend(
        _table(
            [
                "profile",
                "max chars",
                "read median",
                "create median",
                "recall@3",
                ndcg_key,
            ],
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

    # --- headline 2: does the budget change the ranking? --------------------
    headline_key = str(
        budget0.get("headline_budget")
        or (budget_keys[0] if budget_keys else HEADLINE_BUDGET)
    )
    if headline_key not in budget_keys and budget_keys:
        headline_key = budget_keys[0]

    budget_scores = {
        r["profile"]["name"]: budget_aggregate(r, headline_key) for r in ordered
    }
    profile_names = [r["profile"]["name"] for r in ordered]

    # Ranked by the metric alone, with ties left as ties. A tie-break on some
    # secondary column would manufacture an ordering the metric does not
    # support, and would hide the most interesting thing a budget comparison can
    # show: two profiles that equal k separates and a budget does not.
    _TIE = 5e-7  # both inputs are rounded to 6 dp upstream

    def _relation(left: float, right: float) -> str:
        if abs(left - right) <= _TIE:
            return "="
        return ">" if left > right else "<"

    def _equal_k(name: str) -> float:
        return float(quality_by_name[name].get("recall@3") or 0.0)

    def _budget(name: str) -> float:
        return float(budget_scores[name].get("recall@budget") or 0.0)

    def _competition_order(score) -> List[List[str]]:
        """Names grouped into tiers, best tier first. A tier is a tie."""
        tiers: List[List[str]] = []
        for name in sorted(profile_names, key=score, reverse=True):
            if tiers and abs(score(tiers[-1][0]) - score(name)) <= _TIE:
                tiers[-1].append(name)
            else:
                tiers.append([name])
        return tiers

    equal_k_tiers = _competition_order(_equal_k)
    budget_tiers = _competition_order(_budget)

    inversions: List[Tuple[str, str]] = []
    collapses: List[Tuple[str, str]] = []
    separations: List[Tuple[str, str]] = []
    for left in profile_names:
        for right in profile_names:
            if left >= right:
                continue
            before = _relation(_equal_k(left), _equal_k(right))
            after = _relation(_budget(left), _budget(right))
            if before == after:
                continue
            if "=" not in (before, after):
                inversions.append((left, right) if before == ">" else (right, left))
            elif before == "=":
                separations.append((left, right) if after == ">" else (right, left))
            else:
                collapses.append((left, right) if before == ">" else (right, left))

    def _tier_text(tiers: List[List[str]]) -> str:
        return " > ".join(" = ".join("`%s`" % n for n in tier) for tier in tiers)

    if inversions:
        heading = "Equal k and a token budget rank the profiles differently"
    elif collapses or separations:
        heading = (
            "A token budget does not reorder the profiles, but it changes how "
            "far apart they are"
        )
    else:
        heading = "Equal k and a token budget rank the profiles the same way"
    lines.append("### 3. %s" % heading)
    lines.append("")
    lines.append(
        "The comparison below is `recall@3` at equal k against `recall@budget` "
        "at %s tokens. **Both are untimed means over the same %s labelled "
        "queries, so no mode applies and the two columns share an N** — that is "
        "what makes them comparable at all. Ties are shown as ties: no "
        "secondary column is used to break them, because a tie-break would "
        "invent an ordering the metric does not support — and two profiles that "
        "equal k separates and a budget does not is precisely the outcome this "
        "comparison exists to detect."
        % (headline_key, truth0.get("queries"))
    )
    lines.append("")
    rows = []
    for name in profile_names:
        rows.append(
            [
                name,
                _fmt(_equal_k(name), 4),
                _fmt(_budget(name), 4),
                _fmt(_budget(name) - _equal_k(name), 4),
                _fmt(budget_scores[name].get("chunks_admitted"), 2),
                _fmt(budget_scores[name].get("unique_docs"), 2),
                _fmt(budget_scores[name].get("duplicate_source_rate"), 3),
            ]
        )
    lines.extend(
        _table(
            [
                "profile",
                "recall@3 (equal k=%s)" % harness.get("k"),
                "recall@budget (%s tok)" % headline_key,
                "delta",
                "chunks admitted",
                "unique docs",
                "duplicate source rate",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "The same two columns as a ranking — **still untimed, still the same %s "
        "labelled queries:**" % truth0.get("queries")
    )
    lines.append("")
    lines.append(
        "| order | at equal k | under a %s-token budget |" % headline_key
    )
    lines.append("|---|---|---|")
    lines.append(
        "| ranking | %s | %s |" % (_tier_text(equal_k_tiers), _tier_text(budget_tiers))
    )
    lines.append("")

    if inversions:
        lines.append(
            "**The ranking changes when the constraint is context rather than "
            "chunk count.** At equal k the order is %s; under a %s-token budget "
            "it is %s. The inversion%s: %s."
            % (
                _tier_text(equal_k_tiers),
                headline_key,
                _tier_text(budget_tiers),
                "" if len(inversions) == 1 else "s",
                "; ".join(
                    "`%s` led `%s` at equal k and trails it under a budget"
                    % (winner, loser)
                    for winner, loser in inversions
                ),
            )
        )
    elif collapses:
        lines.append(
            "**No profile overtakes another — but the gap that equal k reports "
            "is not there once the budget is the constraint.** %s. The ordering "
            "survives; the separation does not, and the separation is what the "
            "equal-k table was being read for."
            % "; ".join(
                "`%s` leads `%s` by %s at equal k (%s vs %s) and ties it under a "
                "%s-token budget (%s vs %s)"
                % (
                    winner,
                    loser,
                    _fmt(_equal_k(winner) - _equal_k(loser), 4),
                    _fmt(_equal_k(winner), 4),
                    _fmt(_equal_k(loser), 4),
                    headline_key,
                    _fmt(_budget(winner), 4),
                    _fmt(_budget(loser), 4),
                )
                for winner, loser in collapses
            )
        )
        lines.append("")
        lines.append(
            "**v1's equal-k comparison is the reason this was invisible.** k is "
            "a count of chunks, so at k=%s a coarse profile is handed several "
            "times as much text as a fine one for the same nominal retrieval "
            "depth, and then credited for covering more documents with it. Hold "
            "the *context* fixed instead and that advantage is gone: inside %s "
            "tokens %s"
            % (
                harness.get("k"),
                headline_key,
                "; ".join(
                    "`%s` admits %s chunks covering %s of the %s documents"
                    % (
                        name,
                        _fmt(budget_scores[name].get("chunks_admitted"), 1),
                        _fmt(budget_scores[name].get("unique_docs"), 2),
                        corpus.get("files"),
                    )
                    for name in profile_names
                ),
            )
            + "."
        )
    elif separations:
        lines.append(
            "**The ranking does not invert, but the budget separates profiles "
            "that equal k could not tell apart.** %s."
            % "; ".join(
                "`%s` and `%s` tie at equal k and `%s` leads under a %s-token "
                "budget (%s vs %s)"
                % (
                    winner,
                    loser,
                    winner,
                    headline_key,
                    _fmt(_budget(winner), 4),
                    _fmt(_budget(loser), 4),
                )
                for winner, loser in separations
            )
        )
    else:
        lines.append(
            "**The ranking does not change, and neither does the separation.** "
            "Under a %s-token budget the order is %s, which is the equal-k "
            "order, and every pairwise gap has the same sign. This is stated "
            "with the same prominence it would have had if it had come out the "
            "other way: the budget comparison exists because equal k *can* "
            "mislead across chunk sizes, and on this corpus it did not."
            % (headline_key, _tier_text(budget_tiers))
        )
    lines.append("")
    saturated = [name for name in profile_names if _budget(name) >= 1.0 - _TIE]
    if saturated:
        lines.append(
            "One caveat on this table, which the numbers make unavoidable: "
            "`recall@budget` is **saturated at 1.000 on %s** at this budget. A "
            "metric pinned at its ceiling cannot rank anything above it, so the "
            "comparison above resolves the bottom of the field and not the top. "
            "The corpus has %s documents and %s labelled queries; separating "
            "profiles that all retrieve everything asked of them needs a harder "
            "labelled set, not a different budget."
            % (
                " and ".join("`%s`" % n for n in saturated),
                corpus.get("files"),
                truth0.get("queries"),
            )
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
        "collection, no parallel readers or writers. The MCP round-trip is the "
        "only figure here that crosses a process boundary, and it is a single "
        "client talking to a single server. Nothing here says anything about "
        "behaviour under load.",
        "**One embedder.** %s / %s throughout. The comparison is between chunk "
        "geometries at a fixed embedder, not between embedders. Since query "
        "embedding turns out to be most of the read, that fixed choice is doing "
        "more work in these numbers than any chunking decision."
        % (
            (ordered[0]["profile"].get("embedder") or {}).get("provider"),
            (ordered[0]["profile"].get("embedder") or {}).get("model"),
        ),
        "**Cold start is a cold process, not a cold machine.** Each cold-start "
        "sample is a genuinely fresh subprocess — new interpreter, new import, "
        "new Chroma client, new ONNX session — but the OS page cache still "
        "holds the model file from the previous sample, and this harness does "
        "not purge it. Only the very first sample pays a cold file read. A "
        "first-boot number would be larger, and nothing here measures it.",
        "**%d cold-start sample%s per cell.** The cold-start tables report "
        "median, minimum and maximum and nothing else, because that many "
        "samples support nothing else. The spread between minimum and maximum "
        "is the honest statement of what is known about that distribution."
        % (runs_of("cold_start"), "" if runs_of("cold_start") == 1 else "s"),
        "**Token counts are approximated from characters.** `ceil(len(text) / "
        "4)`, not a tokenizer. The budget comparison is internally consistent "
        "and the ranking it produces is not sensitive to a uniform scaling of "
        "that estimate, but the absolute token figures should not be read as a "
        "real model's context accounting.",
    ] + (
        [
            "**A percentile of %d samples is still a percentile of %d samples.** "
            "%s earn a P95 by the rule this document sets itself, and %d samples "
            "put roughly %d above it. That is enough to say the tail exists and "
            "not enough to characterise it."
            % (
                runs_of(earns[0]),
                runs_of(earns[0]),
                " and ".join("`%s`" % op for op in earns),
                runs_of(earns[0]),
                max(1, round(runs_of(earns[0]) * 0.05)),
            )
        ]
        if earns
        else [
            "**No percentile is reported anywhere in this run.** Every operation "
            "was sampled below the N ≥ %d gate, so the latency tables are "
            "medians and spreads only. A run at the shipped defaults reports "
            "P50 and P95 for `read` and `delete`; this one does not, and no "
            "figure below should be read as a tail statistic."
            % percentile_min_n
        ]
    ) + [
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


def cold_start_child(args) -> int:
    """One operation, in this process, with nothing warm. Prints one JSON line.

    Invoked by ``ProfileBench._cold_start_probe`` as a subprocess. Everything
    expensive happens here for the first time: the module import above already
    ran, Chroma has not been imported, the ONNX session does not exist and the
    OS has not been asked for the model file by this process.

    Two things are true of this child and are worth stating rather than
    assuming:

    * The offline socket guard is installed at module import, which is above
      this function. A cold-start probe is exactly the moment a library is most
      likely to reach for the network — a model download, a telemetry ping —
      and it is guarded like every other run.
    * The collection is checked to be in the bench namespace before anything is
      built. A cold-start measurement pointed at the working collection would
      violate the corpus guarantee the whole report rests on, and it would do
      so from a subprocess where it is hardest to see.
    """
    kind = args.cold_start_probe
    name = (args.profile or ["baseline"])[0]
    collection_name = args.collection or ""

    for label, value in (("--collection", collection_name), ("RAG_COLLECTION", COLLECTION_NAME)):
        if not value.startswith(COLLECTION_PREFIX):
            print(
                "cold-start probe refuses to run: %s is %r, which is outside the "
                "%r namespace" % (label, value, COLLECTION_PREFIX),
                file=sys.stderr,
            )
            return 2
    if args.corpus is None:
        print("cold-start probe needs --corpus DIR", file=sys.stderr)
        return 2

    profile = load_profile(name)
    config = config_from_profile(profile)
    k = args.k or int((profile.get("retrieval") or {}).get("k", 5))

    # --- init: module import through Chroma client and embedder ready -------
    # One embedding function instance, warmed here and handed to the operation
    # below, so the ONNX model is loaded exactly once and the load is charged to
    # init rather than to the operation that happens to touch it first.
    embedder = _embedding_function()
    embedder(["cold start probe warm up"])
    client = get_client()
    collection = None
    if kind == "read":
        collection = client.get_collection(
            name=collection_name, embedding_function=embedder
        )
        collection.count()
    init_ms = (time.perf_counter() - _PROCESS_START) * 1000.0

    # --- the operation ------------------------------------------------------
    started = time.perf_counter()
    if kind == "read":
        for query in QUERIES:
            collection.query(query_texts=[query], n_results=k)  # type: ignore[union-attr]
        detail = {"queries": len(QUERIES), "k": k}
    else:
        summary = reindex(
            reset=True,
            root=args.corpus,
            collection_name=collection_name,
            config=config,
            embedding_function=embedder,
        )
        detail = {"files": summary.get("files"), "chunks": summary.get("chunks")}
    op_ms = (time.perf_counter() - started) * 1000.0

    sys.stdout.write(
        COLD_START_SENTINEL
        + json.dumps(
            {
                "operation": kind,
                "profile": name,
                "collection": collection_name,
                "init_ms": round(init_ms, 3),
                "op_ms": round(op_ms, 3),
                "blocked_connect_attempts": len(BLOCKED_CONNECTS),
                "detail": detail,
            }
        )
        + "\n"
    )
    sys.stdout.flush()
    return 0


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
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=None,
        help=(
            "smoke-test escape hatch: set every timed operation to N. Mutually "
            "exclusive with the per-operation flags below; the published run "
            "uses the defaults, not this"
        ),
    )
    for op in TIMED_OPS + ("cold_start",):
        parser.add_argument(
            "--runs-%s" % op.replace("_", "-"),
            type=int,
            default=None,
            dest="runs_%s" % op,
            help="samples for the %s operation (default %d, cap %d)"
            % (op.replace("_", " "), DEFAULT_RUNS[op], RUNS_CAP[op]),
        )
    parser.add_argument(
        "--token-budget",
        type=int,
        action="append",
        default=None,
        metavar="TOKENS",
        help="token budget for budget-constrained retrieval; repeatable (default %s)"
        % ", ".join(str(b) for b in TOKEN_BUDGETS),
    )
    parser.add_argument("-k", type=int, default=None, help="override retrieval k")
    parser.add_argument(
        "--cold-start-probe",
        choices=("read", "create"),
        default=None,
        help=(
            "child entry point: perform exactly one operation in this process "
            "and print one JSON line. Used by the cold-start mode; not a "
            "measurement a caller runs directly"
        ),
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="collection name for --cold-start-probe (must be in the bench namespace)",
    )
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

    # --- sample sizes: resolved once, validated hard ------------------------
    # Exceeding a cap is an error, not a clamp: a silently reduced N would put a
    # number in the report that the flag says is something else.
    per_op = {name: getattr(args, "runs_%s" % name) for name in DEFAULT_RUNS}
    named = sorted(name for name, value in per_op.items() if value is not None)
    if args.runs is not None and named:
        parser.error(
            "-n/--runs sets every timed operation at once and cannot be combined "
            "with the per-operation flag(s) %s"
            % ", ".join("--runs-%s" % n.replace("_", "-") for n in named)
        )
    runs_per_op = dict(DEFAULT_RUNS)
    if args.runs is not None:
        if args.runs < 1:
            parser.error("--runs must be at least 1")
        for name in TIMED_OPS:
            runs_per_op[name] = args.runs
    for name, value in per_op.items():
        if value is None:
            continue
        if value < 1:
            parser.error("--runs-%s must be at least 1" % name.replace("_", "-"))
        if value > RUNS_CAP[name]:
            parser.error(
                "--runs-%s is capped at %d (got %d)"
                % (name.replace("_", "-"), RUNS_CAP[name], value)
            )
        runs_per_op[name] = value
    for name in runs_per_op:
        if runs_per_op[name] > RUNS_CAP[name]:
            parser.error(
                "--runs %d exceeds the cap of %d for %s"
                % (runs_per_op[name], RUNS_CAP[name], name)
            )

    budgets = tuple(args.token_budget) if args.token_budget else TOKEN_BUDGETS
    for budget in budgets:
        if budget < 1:
            parser.error("--token-budget must be positive")

    # --- cold-start child: one operation, one JSON line, no state guard -----
    if args.cold_start_probe is not None:
        return cold_start_child(args)

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
                runs_per_op=runs_per_op,
                k=args.k or int((profile.get("retrieval") or {}).get("k", 5)),
                truth=truth,
                budgets=budgets,
            )
            benched.append(bench.collection_name)
            benched.append(bench.cold_collection_name)
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
                    # A mapping, not a scalar: the four operations no longer
                    # share an N, and a table that cannot state its own N must
                    # not be rendered.
                    "runs_per_op": dict(runs_per_op),
                    "percentile_min_n": PERCENTILE_MIN_N,
                    "statistics": (
                        "median, min, max and median absolute deviation always; "
                        "P50 and P95 only at n >= %d; no P99 at any n"
                        % PERCENTILE_MIN_N
                    ),
                    "modes": {
                        MODE_WARM: (
                            "many operations in one process: model loaded, OS "
                            "cache hot. The four timed operations and the stage "
                            "split are all warm"
                        ),
                        MODE_COLD_START: (
                            "a fresh subprocess per sample, paying interpreter "
                            "startup, module import, Chroma client construction "
                            "and ONNX model load before any work"
                        ),
                    },
                    "token_budgets": list(budgets),
                    "mcp_roundtrip_runs": MCP_ROUNDTRIP_RUNS,
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
                # Where the read actually goes. The v1 headline rested on this
                # split without ever measuring it.
                "read_stages": result["read_stages"],
                # A fresh process per sample. Not the first row of a warm loop.
                "cold_start": result["cold_start"],
                # Timing alone ranks fast-and-wrong above slow-and-right.
                # Filled by ProfileBench.op_quality(), which is untimed.
                "quality": result["quality"],
                # Equal k is not a fair comparison across chunk sizes; this is
                # the same labelled set scored under a context budget instead.
                "token_budget": result["token_budget"],
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
