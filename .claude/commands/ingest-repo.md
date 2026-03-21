# Ingest a GitHub Repository

Ingest the repository at the given URL into the vector index.

## Usage
```
/ingest-repo <github-url>
```

## What this does
1. Clones or fetches the repo via GitHub API
2. Filters files (skips binaries, lock files, node_modules, etc.)
3. Chunks code by AST boundaries (functions/classes)
4. Embeds chunks with nomic-embed-code
5. Upserts into Qdrant Cloud collection

## Steps

Run the ingestion pipeline for the provided GitHub URL: $ARGUMENTS

- Call `ingestion/repo_fetcher.py` to fetch the repo
- Call `ingestion/file_filter.py` to get the list of files to index
- Call `ingestion/code_chunker.py` to chunk each file
- Call `backend/services/ingestion_service.py` to embed and store
- Print a summary: files indexed, chunks stored, languages detected
