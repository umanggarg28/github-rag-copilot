# nanochat

A minimal GPT implementation and execution engine for evaluating language model tasks.

## What it does

Evaluates LLM performance across structured benchmarks using `Task` and `TaskSequence` abstractions. It pairs a `GPT` model with an `Engine` that manages `KVCache` and `RowState` to generate tokens and execute code within a controlled environment.

## Architecture

*   `gpt.py`: Defines the core transformer architecture, including `GPT`, `MLP`, and `Linear` layers.
*   `engine.py`: Handles the inference loop, token sampling, and state management.
*   `execution.py`: Provides a sandbox for running generated code using `capture_io` and `reliability_guard`.
*   `tokenizer.py`: Implements tokenization via `HuggingFaceTokenizer` and `RustBPETokenizer`.
*   `common.py` & `report.py`: Manage task mixtures and system performance reporting.

## Key Components

*   `GPT` — The primary transformer model implementation.
*   `Engine` — Manages the generation process and token sampling.
*   `KVCache` — Optimizes inference by caching key-value pairs for previous tokens.
*   `TaskSequence` — Organizes and executes a series of evaluation tasks.
*   `capture_io` — Executes code and captures standard output and error for evaluation.
*   `checkpoint_manager.py` — Provides utilities to load and save model weights.

## Usage

```python
from gpt import GPT
from engine import Engine
from tokenizer import get_tokenizer
from checkpoint_manager import load_model

# Initialize tokenizer and model
tokenizer = get_tokenizer("path/to/tokenizer")
model = load_model("path/to/checkpoint")

# Setup engine for generation
engine = Engine(model, tokenizer)
output = engine.sample_next_token("User: Hello\nAssistant:")
```

## Tech Stack

*   Python
*   PyTorch (via `Linear`, `MLP`, and `GPT` implementations)
*   HuggingFace Tokenizers
*   WandB (integration via `DummyWandb`)