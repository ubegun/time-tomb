# Time Tomb

A Time Tomb for a knowledge base: the repo is public proof the knowledge
exists and when; the body stays sealed on the owner's machine.

A local-first RAG (retrieval-augmented generation) toolkit with a split
publishing model: the **skeleton** — a salted Merkle manifest plus all
tooling — is public; the **body** — markdown notes, vector embeddings,
salt, and name resolver — exists only on the owner's workstation.

What an external reader CAN derive: the number of chunks/files, the tree
shape, and the fact and time of revisions.
What they CANNOT derive: names, content, or embeddings.

## Stack

Python 3.9+, ChromaDB with local ONNX embeddings (fully offline),
MCP server (JSON-RPC over stdio) for AI-agent access.

## Integrity model

Every content revision changes the manifest root hash, so the commit
history is an auditable timeline of the knowledge base — integrity proof
without content exposure. Any tampering with the local body is detectable
by recomputing the manifest and comparing root hashes.

## Benchmarks

Read latency is dominated by query embedding rather than by index size, so it
is effectively flat across a wide span of chunk sizes and timing alone selects
the wrong chunk profile; `recall@3` and `nDCG@10` are what pick the shipped
default. The numbers, the method and the limitations are in
[BENCHMARKS.md](BENCHMARKS.md) — generated from the result files, never typed.

Regenerate all of it from a fresh clone:

    ./install.sh && python bench.py --all --corpus bench/corpus --corpus-quality bench/corpus-quality

That runs the three chunk profiles against **two** deterministic synthetic
corpora, both committed to this repository together with their labelled query
sets. `bench/corpus/` is the timed one: every latency, cold-start and
stage-split figure is a property of its seven documents, so it is frozen and
never grown. `bench/corpus-quality/` is larger — 24 documents — and is read only
by the untimed scoring, because the 64-query labelled set needs more planting
slots than the timed corpus has and growing that one would move every latency
number for no reason. Each operation has its own sample count — read and delete 100, update 50,
create 20, cold start 5 — and the statistics are gated on it: a percentile is
rendered only where the sample count supports one, and no P99 is computed
anywhere. Latency is reported in two named modes, `warm` (many operations in
one process) and `cold-start` (a fresh subprocess per sample), and the read is
broken down into query embedding, vector search, materialization and the full
MCP round-trip. Retrieval is scored both at equal k and under a fixed token
budget, because equal k is not a fair comparison across chunk sizes.

The labelled set is 64 English queries, and every one of them declares a
**category** (`single-hop`, `multi-doc`, `vague`, `unanswerable`), a **split**
(`held-out` or `tuning`) and a **phrasing** (`natural`, `terse`, `verbose`,
`misspelled`). Scores are reported per category and per split, and the headline
comparison is computed on the held-out subset only — the three profiles already
existed when the set was written, so a set that is entirely visible to profile
selection cannot support a claim about profile selection. Eight unanswerable
controls must score exactly zero on every rung and under every budget or the
run fails and writes nothing. A separate multilingual probe measures what the
shipped English embedder does with German, Russian and Ukrainian queries; it is
reported on its own and never enters a ranking.

The corpus on disk is checked byte-for-byte against its generator before
anything is measured, and `BENCHMARKS.md` is rewritten from the result files at
the end, so `git diff` shows exactly how your machine differs from the
published run. No knowledge base is needed: the benchmarks bring their own
corpus, and a clone without one installs and runs fine. The full run takes
several minutes; `-n N` is a smoke-test escape hatch and its output is not a
publishable run.

## Layout (published part only)

| Path | Purpose |
|---|---|
| `manifest.json` | Merkle skeleton: root / tree / leaves (sha256[:16]) |
| `index_toolbox.py` | chunk markdown notes → local Chroma collection |
| `search.py` | CLI semantic search |
| `manifest.py` | build manifest, `--diff` changed subtrees |
| `mcp_server.py` | MCP server exposing search/reindex to AI agents |
| `publish_skeleton.py` | rebuild the publishing sandbox (skeleton only) |
| `raglog.py` | shared logging |
| `install.sh` | bootstrap: virtualenv, dependencies, index, manifest, MCP check |
| `requirements.txt` | pinned dependencies, exactly as measured |
| `bench.py` | CRUD timing + retrieval-quality harness; also emits the corpus |
| `bench/corpus/` | the timed synthetic corpus and `ground-truth.json`, byte-reproducible and frozen |
| `bench/corpus-quality/` | the larger untimed corpus the 64-query labelled set is planted into |
| `bench/results/` | one result file per profile per run |
| `BENCHMARKS.md` | generated from `bench/results/` |
| `profiles/` | the three chunk profiles the benchmarks compare |

Every row above is a file or directory that exists in this repository. Nothing
else is published: the publisher works from an allowlist, so a file added
beside these stays unpublished until it is named.

Never in this repo: the salt, the name resolver map, note bodies,
embeddings, vector DB, logs.

## Bounded sources

What may be read is declared, not discovered. A retrieval profile may name a
*source* — the directories to walk and an allowlist bounding them — and a
profile that names none is treated as declaring the built-in default. Every
candidate path is resolved (symlinks followed, `..` collapsed) and compared
against the allowlist by path components, so a neighbouring directory whose
name merely starts with an allowed one is not admitted. A path that escapes
fails the whole run instead of being skipped: a corpus silently missing part
of itself is worse than a refusal.

## Planned, not yet built

- Encrypted machine-to-machine transport (PBKDF2 + AES) for moving a body
  between the owner's own machines.

## License

Apache-2.0.

Maintainer: [ubegun](https://github.com/ubegun). Tooling is self-contained
and reusable; adapting to another knowledge base requires only changing
the root path.
