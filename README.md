# 🤖 docs-agent

A zero-infrastructure GitHub documentation bot. Drop it into any repo and let users
query your docs directly from issues.

```
/ask How do I install this?
```

→ The bot replies with a sourced answer in seconds.

## How it works

```
docs/*.md  ──push──▶  [Index Action]  ──▶  docs_index.json (committed)
                       all-MiniLM-L6-v2             │
                       runs locally, free            │
                                                     │
issue comment "/ask ..."  ──▶  [Agent Action]  ──────┘
                                embed query (local, free)
                                → cosine sim → top chunks
                                → Claude Sonnet → reply comment
                                     ▼
                              @user gets an answer
```

## Quick Start

1. **Fork or copy** this repo into your project (or just copy `.github/` and `scripts/`)
2. **Add ONE secret** in Settings → Secrets → Actions:
   - `ANTHROPIC_API_KEY` ← only key you need
3. **Put your docs** in `docs/` as `.md` files
4. **Run** the "Index Documentation" workflow once manually
5. **Try it**: open an issue and comment `/ask anything about your docs`

Embeddings use `all-MiniLM-L6-v2` running locally inside GitHub Actions — no OpenAI key,
no embedding cost ever.

See [`docs/README.md`](docs/README.md) for full configuration options.

## Cost

- **Indexing**: free (local embeddings, no API)
- **Answering**: ~$0.002–0.01 per question (Claude Sonnet only)

## File structure

```
.github/
  workflows/
    index-docs.yml    ← runs on push to docs/, builds the index locally
    docs-agent.yml    ← runs on /ask comments, replies with sourced answer
scripts/
  build_index.py      ← chunks + embeds docs with sentence-transformers
  answer.py           ← retrieves context, calls Claude, posts reply
docs/                 ← your documentation lives here
docs_index.json       ← auto-generated, committed by the index action
```
