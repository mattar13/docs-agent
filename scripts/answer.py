"""
answer.py — Called by the GitHub Actions comment bot workflow.

Reads env vars set by the workflow, retrieves relevant doc chunks via local
sentence-transformers embeddings, calls Claude for the answer, then posts the
reply back to the GitHub issue. No OpenAI key needed.

Environment variables expected:
    ANTHROPIC_API_KEY   — for generating the answer via Claude
    GH_TOKEN            — GitHub token to post comment (GITHUB_TOKEN is fine)
    QUESTION            — raw comment body, e.g. "/ask How do I install this?"
    ISSUE_COMMENTS_URL  — GitHub API URL to post reply to
    COMMENT_USER        — GitHub username who asked
    INDEX_FILE          — (optional) path to docs_index.json
    EMBED_MODEL         — (optional) HuggingFace model name, default all-MiniLM-L6-v2
"""

import json
import os
import sys
import urllib.request
import urllib.error

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from anthropic import Anthropic
except ImportError:
    print("Install: pip install sentence-transformers anthropic numpy")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_FILE  = os.getenv("INDEX_FILE", "docs_index.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K       = int(os.getenv("TOP_K", "5"))
TRIGGER     = os.getenv("TRIGGER", "/ask")
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "1024"))
# ─────────────────────────────────────────────────────────────────────────────


def load_index(path: str) -> tuple[list[dict], np.ndarray, bool]:
    with open(path, "r") as f:
        data = json.load(f)
    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"], dtype="float32")
    normalized = data.get("normalized", False)
    return chunks, embeddings, normalized


def embed_query(query: str, model: SentenceTransformer, normalized: bool) -> np.ndarray:
    """Embed the query, matching the normalization used at index time."""
    return model.encode(
        [query],
        normalize_embeddings=normalized,
        convert_to_numpy=True,
    )[0].astype("float32")


def retrieve(query: str, chunks: list[dict], embeddings: np.ndarray,
             model: SentenceTransformer, normalized: bool, top_k: int) -> list[dict]:
    q_emb = embed_query(query, model, normalized)
    if normalized:
        # Both sides are unit vectors; dot product == cosine similarity
        sims = embeddings @ q_emb
    else:
        # Compute cosine similarity manually
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        e_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        sims = e_norm @ q_norm

    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(sims[idx])
        results.append(chunk)
    return results


def build_answer(question: str, context_chunks: list[dict], anth_client: Anthropic) -> str:
    context_parts = []
    for i, c in enumerate(context_chunks, 1):
        context_parts.append(f"[Source {i}: {c['source']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    response = anth_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=MAX_TOKENS,
        system=(
            "You are a helpful documentation assistant for a GitHub repository. "
            "Answer the user's question using ONLY the documentation context provided. "
            "If the answer is not in the context, say so honestly — don't invent information. "
            "Keep answers concise and technical. Use markdown formatting."
        ),
        messages=[{
            "role": "user",
            "content": f"Documentation context:\n\n{context}\n\n---\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text


def post_github_comment(comments_url: str, body: str, token: str) -> None:
    payload = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(
        comments_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "docs-agent-bot",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"Comment posted: HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        print(f"Failed to post comment: {e.code} {e.read().decode()}")
        sys.exit(1)


def extract_question(comment_body: str, trigger: str) -> str | None:
    body = comment_body.strip()
    if not body.lower().startswith(trigger.lower()):
        return None
    question = body[len(trigger):].strip()
    return question if question else None


def format_reply(question: str, answer: str, top_chunks: list[dict], user: str) -> str:
    sources = "\n".join(
        f"  - `{c['source']}` (score: {c['score']:.2f})" for c in top_chunks
    )
    return (
        f"Hey @{user}! 🤖 Here's what I found in the docs:\n\n"
        f"**Q: {question}**\n\n"
        f"{answer}\n\n"
        f"<details>\n<summary>Sources consulted</summary>\n\n{sources}\n\n</details>\n\n"
        f"---\n*Powered by docs-agent · [How it works](.github/workflows/docs-agent.yml)*"
    )


def main():
    raw_comment  = os.environ.get("QUESTION", "")
    comments_url = os.environ.get("ISSUE_COMMENTS_URL", "")
    gh_token     = os.environ.get("GH_TOKEN", "")
    comment_user = os.environ.get("COMMENT_USER", "user")

    question = extract_question(raw_comment, TRIGGER)
    if not question:
        print(f"Comment does not start with '{TRIGGER}'. Nothing to do.")
        sys.exit(0)

    print(f"Question: {question}")

    # Check index exists
    if not os.path.exists(INDEX_FILE):
        msg = (
            f"Hey @{comment_user}, the docs index hasn't been built yet. "
            f"Please run the **Index Documentation** workflow manually from the Actions tab, "
            f"or push any change to the `docs/` folder to trigger it automatically."
        )
        post_github_comment(comments_url, msg, gh_token)
        sys.exit(0)

    # Load index + local embedding model
    chunks, embeddings, normalized = load_index(INDEX_FILE)
    print(f"Loaded index: {len(chunks)} chunks (normalized={normalized})")

    print(f"Loading embedding model '{EMBED_MODEL}' ...")
    model = SentenceTransformer(EMBED_MODEL)

    # Retrieve relevant chunks
    top_chunks = retrieve(question, chunks, embeddings, model, normalized, TOP_K)
    print(f"Top chunk scores: {[round(c['score'], 3) for c in top_chunks]}")

    # Generate answer via Claude
    anth_client = Anthropic()
    answer = build_answer(question, top_chunks, anth_client)
    reply  = format_reply(question, answer, top_chunks, comment_user)

    post_github_comment(comments_url, reply, gh_token)
    print("Done.")


if __name__ == "__main__":
    main()
