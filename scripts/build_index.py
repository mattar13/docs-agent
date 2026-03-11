"""
build_index.py — Chunk and embed all docs into a searchable JSON index.
Uses sentence-transformers locally — no API key or cost required.

Run via GitHub Actions whenever docs/ changes, or locally with:
    python scripts/build_index.py
"""

import json
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install dependencies: pip install sentence-transformers numpy")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR      = os.getenv("DOCS_DIR", "docs/")
OUTPUT_FILE   = os.getenv("INDEX_FILE", "docs_index.json")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "400"))    # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # word overlap
EMBED_MODEL   = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # fast, 384-dim, free
EMBED_BATCH   = int(os.getenv("EMBED_BATCH", "64"))
# ─────────────────────────────────────────────────────────────────────────────


def find_doc_files(docs_dir: str) -> list[Path]:
    patterns = ["**/*.md", "**/*.mdx", "**/*.rst", "**/*.txt"]
    files = []
    for pat in patterns:
        files.extend(Path(docs_dir).glob(pat))
    return sorted(set(files))


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[dict]:
    """Sliding-window word-level chunker with overlap."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, max(1, len(words)), step):
        window = words[i : i + chunk_size]
        if not window:
            break
        body = " ".join(window)
        chunk_id = hashlib.md5(f"{source}:{i}:{body}".encode()).hexdigest()[:12]
        chunks.append({
            "id": chunk_id,
            "source": source,
            "text": body,
            "word_start": i,
        })
        if i + chunk_size >= len(words):
            break
    return chunks


def embed_chunks(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode all texts locally in batches. Returns (N, D) float32 array."""
    print(f"  Encoding {len(texts)} chunks in batches of {EMBED_BATCH} ...")
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,  # pre-normalize for fast cosine via dot product
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def main():
    # Load local model (downloaded once, cached by HuggingFace)
    print(f"Loading embedding model '{EMBED_MODEL}' ...")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Find and chunk docs
    doc_files = find_doc_files(DOCS_DIR)
    if not doc_files:
        print(f"No doc files found in '{DOCS_DIR}'. Exiting.")
        sys.exit(0)

    print(f"\nFound {len(doc_files)} doc file(s) in '{DOCS_DIR}'")
    all_chunks = []
    for path in doc_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, str(path), CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        print(f"  {path}: {len(chunks)} chunk(s)")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Embed
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_chunks(texts, model)

    # Write index
    index = {
        "model": EMBED_MODEL,
        "normalized": True,  # embeddings are pre-normalized; use dot product for similarity
        "chunks": all_chunks,
        "embeddings": embeddings.tolist(),
    }
    Path(OUTPUT_FILE).write_text(json.dumps(index, separators=(",", ":")))
    size_kb = Path(OUTPUT_FILE).stat().st_size / 1024
    print(f"\n✅ Index written to '{OUTPUT_FILE}' ({size_kb:.1f} KB, {len(all_chunks)} chunks)")


if __name__ == "__main__":
    main()
