"""
build_index.py — Chunk and embed all docs into a searchable JSON index.
Run via GitHub Actions whenever docs/ changes, or locally with:
    OPENAI_API_KEY=sk-... python scripts/build_index.py
"""

import json
import glob
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("Install dependencies: pip install openai numpy")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR       = os.getenv("DOCS_DIR", "docs/")
OUTPUT_FILE    = os.getenv("INDEX_FILE", "docs_index.json")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "400"))   # words per chunk
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "50")) # word overlap between chunks
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH    = 100  # max texts per API call
# ─────────────────────────────────────────────────────────────────────────────


def find_doc_files(docs_dir: str) -> list[Path]:
    patterns = ["**/*.md", "**/*.mdx", "**/*.rst", "**/*.txt"]
    files = []
    for pat in patterns:
        files.extend(Path(docs_dir).glob(pat))
    return sorted(set(files))


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[dict]:
    """Sliding-window word-level chunker."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, max(1, len(words)), step):
        window = words[i : i + chunk_size]
        if not window:
            break
        chunk_text = " ".join(window)
        chunk_id = hashlib.md5(f"{source}:{i}:{chunk_text}".encode()).hexdigest()[:12]
        chunks.append({
            "id": chunk_id,
            "source": source,
            "text": chunk_text,
            "char_start": i,
        })
        if i + chunk_size >= len(words):
            break
    return chunks


def embed_texts(texts: list[str], client: OpenAI, model: str) -> np.ndarray:
    """Batch-embed a list of strings, returns (N, D) float32 array."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        print(f"  Embedding batch {i // EMBED_BATCH + 1} / {(len(texts) - 1) // EMBED_BATCH + 1} ...")
        response = client.embeddings.create(input=batch, model=model)
        batch_embs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embs)
    return np.array(all_embeddings, dtype="float32")


def main():
    client = OpenAI()  # reads OPENAI_API_KEY from env

    # 1. Find and chunk all doc files
    doc_files = find_doc_files(DOCS_DIR)
    if not doc_files:
        print(f"No doc files found in '{DOCS_DIR}'. Exiting.")
        sys.exit(0)

    print(f"Found {len(doc_files)} doc file(s) in '{DOCS_DIR}'")
    all_chunks = []
    for path in doc_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, str(path), CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        print(f"  {path}: {len(chunks)} chunk(s)")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # 2. Embed all chunks
    print(f"\nEmbedding with model '{EMBED_MODEL}' ...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, client, EMBED_MODEL)

    # 3. Write index
    index = {
        "model": EMBED_MODEL,
        "chunks": all_chunks,
        "embeddings": embeddings.tolist(),
    }
    Path(OUTPUT_FILE).write_text(json.dumps(index, separators=(",", ":")))
    size_kb = Path(OUTPUT_FILE).stat().st_size / 1024
    print(f"\n✅ Index written to '{OUTPUT_FILE}' ({size_kb:.1f} KB, {len(all_chunks)} chunks)")


if __name__ == "__main__":
    main()
