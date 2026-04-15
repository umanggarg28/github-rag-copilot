# Cartographer

Maps and queries codebases using an agentic retrieval system to provide precise technical answers.

## What it does

Cartographer enables deep codebase exploration by combining `RetrievalService` for searching and `GenerationService` for synthesizing answers. It utilizes an MCP server implementation in `mcp_server.py` to provide tools like `search_symbol` and `get_file_chunk`, allowing an LLM agent to navigate files and symbols dynamically.

## Architecture

* `agent.py`: Orchestrates the `AgentService`, managing the loop between LLM tool calls and execution.
* `retrieval.py`: Implements `RetrievalService` and `Reranker` to fetch and prioritize code snippets using `_rrf_merge`.
* `generation.py`: Uses `GenerationService` to classify queries and construct messages for the LLM.
* `mcp_server.py`: Exposes codebase primitives—such as symbol search and note recall—as tools for the agent.
* `schemas.py`: Defines the data contracts for the system, including `AgentRequest`, `SearchResponse`, and `RepoInfo`.

## Key Components

* `AgentService` — Manages the agentic workflow and parses tool calls from LLM responses.
* `RetrievalService` — Handles the retrieval of relevant code segments from the indexed repository.
* `Reranker` — Refines search results to improve the precision of the context provided to the LLM.
* `GenerationService` — Classifies incoming queries and manages the prompt construction process.
* `search_symbol` — Locates specific symbols within the codebase via the MCP server.
* `get_file_chunk` — Retrieves specific segments of a file for detailed analysis.

## Usage

```python
from agent import agent_query
from schemas import AgentRequest

# Initialize a query request
request = AgentRequest(
    query="Explain the implementation of the Reranker in retrieval.py"
)

# Execute the agent query
response = agent_query(request)
print(response)
```

## Tech Stack

* **Language**: Python
* **LLM Integration**: OpenRouter (via `_openrouter_client`)
* **Architecture**: Model Context Protocol (MCP)
* **Retrieval Logic**: Reciprocal Rank Fusion (`_rrf_merge`) and Reranking