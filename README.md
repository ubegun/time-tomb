# rag-skeleton

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

Every row above is a file that exists in this repository. Nothing else is
published.

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
