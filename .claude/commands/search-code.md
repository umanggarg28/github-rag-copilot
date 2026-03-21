# Search Code Without Generating an Answer

Run a retrieval-only search against the indexed repositories.

## Usage
```
/search-code <query>
```

## What this does
Calls the `/search` endpoint (no LLM) and displays the raw retrieved chunks
with their file paths, line numbers, and relevance scores. Useful for:
- Verifying the index contains what you expect
- Debugging retrieval quality before blaming the LLM
- Exploring the codebase without a full RAG query

## Steps

Search for: $ARGUMENTS

Call `GET /search?query=<query>&top_k=10` and display:
- Filepath + line range for each result
- Relevance score
- The actual code chunk
