# docs-agent Documentation

## Overview

docs-agent is a lightweight GitHub-native documentation bot. Users can query your
project's documentation directly from GitHub issues by commenting `/ask <question>`.

## How It Works

1. Whenever you push changes to the `docs/` folder, a GitHub Action runs and
   re-indexes your documentation using OpenAI embeddings.
2. The index is stored as `docs_index.json` in your repository.
3. When a user comments `/ask <question>` on any issue, the bot:
   - Embeds the question
   - Finds the most relevant doc chunks via cosine similarity
   - Sends those chunks + the question to Claude
   - Posts the answer as a follow-up comment

## Installation

### Prerequisites

- An OpenAI API key (for embeddings)
- An Anthropic API key (for answer generation)

### Steps

1. Copy the files from this repo into your project:
   ```
   .github/workflows/index-docs.yml
   .github/workflows/docs-agent.yml
   scripts/build_index.py
   scripts/answer.py
   ```

2. Add your API keys as GitHub repository secrets:
   - Go to **Settings → Secrets and variables → Actions**
   - Add `OPENAI_API_KEY`
   - Add `ANTHROPIC_API_KEY`

3. Make sure your documentation lives in a `docs/` folder (or update `DOCS_DIR`
   in the workflow).

4. Trigger the indexing workflow manually (Actions → Index Documentation → Run workflow),
   or just push any change to `docs/`.

## Configuration

You can customize behavior by editing env vars in the workflow files:

| Variable | Default | Description |
|---|---|---|
| `DOCS_DIR` | `docs/` | Where your markdown files live |
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `1024` | Max tokens in the answer |
| `TRIGGER` | `/ask` | Comment prefix to trigger the bot |

## Usage

In any GitHub issue, comment:

```
/ask How do I install this project?
/ask What configuration options are available?
/ask Does this support Python 3.12?
```

The bot will reply with an answer sourced from your documentation.

## Costs

- **Indexing**: ~$0.001–0.02 per indexing run (depends on doc size), using
  `text-embedding-3-small`
- **Answering**: ~$0.002–0.01 per question using Claude Sonnet

Very cheap for typical OSS documentation volumes.

## Supported File Types

The indexer picks up: `.md`, `.mdx`, `.rst`, `.txt`

## Troubleshooting

**The bot doesn't respond**
- Check that both workflows are enabled (Actions tab)
- Verify your secrets are set correctly
- Check the Actions run log for error messages

**"Docs index hasn't been built yet"**
- Run the "Index Documentation" workflow manually from the Actions tab

**Answers are low quality**
- Try decreasing `CHUNK_SIZE` for more precise retrieval
- Increase `TOP_K` to give the model more context
- Make sure your docs are detailed and well-structured
