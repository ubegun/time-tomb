#!/usr/bin/env python
"""CLI semantic search over the local Chroma collection.

Usage:
    python search.py "your question here" -k 5
    python search.py "another question" --json
"""

import argparse
import json
from typing import Iterable, List, Optional

from raglog import COLLECTION_NAME, get_logger

log = get_logger("search")


def search(query: str, k: int = 5, source: Optional[str] = None) -> List[dict]:
    """Return the k nearest chunks as plain dicts, best match first."""
    from index_toolbox import get_collection

    collection = get_collection()
    total = collection.count()
    if total == 0:
        log.warning("collection '%s' is empty - run index_toolbox.py", COLLECTION_NAME)
        return []

    kwargs = {"query_texts": [query], "n_results": min(k, total)}
    if source:
        kwargs["where"] = {"source": source}

    raw = collection.query(**kwargs)

    hits: List[dict] = []
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]

    for position, document in enumerate(documents):
        metadata = metadatas[position] if position < len(metadatas) else {}
        distance = distances[position] if position < len(distances) else None
        hits.append(
            {
                "rank": position + 1,
                "id": ids[position] if position < len(ids) else None,
                "source": metadata.get("source", "?"),
                "heading": metadata.get("heading", ""),
                "chunk_index": metadata.get("chunk_index"),
                # cosine distance -> similarity, for a number that reads the
                # right way round in the CLI output
                "score": None if distance is None else round(1.0 - float(distance), 4),
                "text": document,
            }
        )

    log.info("query %r -> %d hits (k=%d)", query, len(hits), k)
    return hits


def format_hits(hits: List[dict], width: int = 400) -> str:
    if not hits:
        return "no results"
    blocks = []
    for hit in hits:
        text = hit["text"].strip().replace("\n", " ")
        if len(text) > width:
            text = text[:width].rstrip() + "..."
        header = "[%d] %s" % (hit["rank"], hit["source"])
        if hit["heading"]:
            header += " :: %s" % hit["heading"]
        if hit["score"] is not None:
            header += "  (score %.4f)" % hit["score"]
        blocks.append("%s\n    %s" % (header, text))
    return "\n\n".join(blocks)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    parser.add_argument("--source", default=None, help="restrict to one KB-relative path")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--width", type=int, default=400, help="snippet width")
    args = parser.parse_args(list(argv) if argv is not None else None)

    hits = search(args.query, k=args.k, source=args.source)

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
    else:
        print(format_hits(hits, width=args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
