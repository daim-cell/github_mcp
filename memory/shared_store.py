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

_client = chromadb.PersistentClient(path=str(Path(__file__).parent.parent / ".chromadb"))
_collection = _client.get_or_create_collection("research_findings")


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

    _collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=chunks,
        metadatas=[base_meta] * len(chunks),
    )

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
    results = _collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
    )

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
