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
                                                    │
issue comment "/ask ..."  ──▶  [Agent Action]  ─────┘
                                     │  embed query → cosine sim → top chunks
                                     │  → Claude Sonnet → reply comment
                                     ▼
                              @user gets an answer
```

## Quick Start

1. **Fork or copy** this repo into your project (or just copy `.github/` and `scripts/`)
2. **Add secrets** in Settings → Secrets → Actions:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
3. **Put your docs** in `docs/` as `.md` files
4. **Run** the "Index Documentation" workflow once manually
5. **Try it**: open an issue and comment `/ask anything about your docs`

See [`docs/README.md`](docs/README.md) for full configuration options.

## Cost

Typical OSS project: **< $1/month** for hundreds of questions.

## File structure

```
.github/
  workflows/
    index-docs.yml    ← runs on push to docs/, builds the index
    docs-agent.yml    ← runs on issue comments starting with /ask
scripts/
  build_index.py      ← chunks + embeds docs into docs_index.json
  answer.py           ← retrieves context, calls Claude, posts reply
docs/                 ← your documentation lives here
docs_index.json       ← auto-generated, committed by the index action
```
