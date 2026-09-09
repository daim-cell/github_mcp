import json
import time
import uuid
from pathlib import Path

import chromadb
import tiktoken

_CHUNK_TOKENS = 512
_OVERLAP_TOKENS = 50
_LOG_FILE = Path(__file__).parent.parent / "logs" / "pipeline_runs.jsonl"
_tokenizer = tiktoken.get_encoding("cl100k_base")

_DB_PATH = str(Path(__file__).parent.parent / ".chromadb")

# Use the Chroma HTTP server if running (port 8000), otherwise fall back to
# PersistentClient. The HTTP server is a single process that owns the HNSW
# index exclusively, so concurrent reads/writes across agent processes are safe.
# Start it with: chroma run --path .chromadb --port 8010
_CHROMA_HTTP_HOST = "127.0.0.1"
_CHROMA_HTTP_PORT = 8010


def _get_collection():
    """
    Return a Chroma collection handle. Prefers the HTTP server (safe for
    multi-process use); falls back to PersistentClient when the server is
    not running.
    """
    try:
        client = chromadb.HttpClient(host=_CHROMA_HTTP_HOST, port=_CHROMA_HTTP_PORT)
        client.heartbeat()  # raises if server not reachable
        return client.get_or_create_collection("research_findings")
    except Exception:
        # HTTP server not running — use PersistentClient (single-process only)
        client = chromadb.PersistentClient(path=_DB_PATH)
        return client.get_or_create_collection("research_findings")


def _chunk(text: str) -> list[str]:
    """Split text into overlapping token-bounded chunks."""
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + _CHUNK_TOKENS, len(tokens))
        chunks.append(_tokenizer.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - _OVERLAP_TOKENS
    return chunks or [text]


def write_findings(
    topic: str, question: str, content: str, metadata: dict = {}
) -> int:
    """Chunk content and store in ChromaDB. Returns number of chunks written."""
    if not content.strip():
        return 0

    chunks = _chunk(content)
    base_meta = {"topic": topic, "question": question, **metadata}

    try:
        col = _get_collection()
        col.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=chunks,
            metadatas=[base_meta] * len(chunks),
        )
    except Exception as e:
        print(f"  [shared_store] write failed ({e}), findings not persisted", flush=True)
        return 0

    record = {
        "timestamp": time.time(),
        "topic": topic,
        "question": question,
        "chunks_written": len(chunks),
    }
    _LOG_FILE.parent.mkdir(exist_ok=True)
    with _LOG_FILE.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    return len(chunks)


def retrieve_relevant(
    query: str, topic: str = "", top_k: int = 5
) -> list[dict]:
    """Retrieve top_k chunks relevant to query, optionally filtered by topic."""
    where = {"topic": topic} if topic else None

    try:
        col = _get_collection()
        filtered_count = len(col.get(where=where)["ids"]) if where else col.count()
        if filtered_count == 0:
            return []
        n = min(top_k, filtered_count)
        results = col.query(query_texts=[query], n_results=n, where=where)
    except Exception as e:
        print(f"  [shared_store] query failed ({e}), returning empty results", flush=True)
        return []

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "content": doc,
            "topic": meta.get("topic", ""),
            "question": meta.get("question", ""),
            "similarity_score": round(1 - dist, 4),
        })

    return output
