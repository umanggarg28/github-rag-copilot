# nanoGPT

Minimalist GPT implementation for training and evaluating small-scale language models.

## What it does

Implements a decoder-only transformer architecture for causal language modeling. It provides a streamlined pipeline to tokenize data via `prepare.py`, train models using `train.py`, and evaluate results through `eval_gpt2.py`.

## Architecture

* `model.py`: Defines the core transformer architecture, including the `GPT` class and its constituent layers.
* `train.py`: Manages the training loop, batching via `get_batch`, and learning rate scheduling via `get_lr`.
* `prepare.py`: Handles dataset preprocessing using `encode` and `decode` functions.
* `eval_gpt2.py` and `eval_gpt2_xl.py`: Provide scripts for model evaluation and benchmarking.

## Key Components

* `GPT` — The primary transformer model class.
* `Block` — A single transformer layer combining attention and feed-forward networks.
* `CausalSelfAttention` — Implements masked self-attention for autoregressive prediction.
* `MLP` — The multi-layer perceptron used within each transformer block.
* `GPTConfig` — Holds hyperparameters for model configuration.
* `get_batch` — Samples training data batches for the model.

## Usage

```python
from model import GPT, GPTConfig

# Initialize model configuration and instance
config = GPTConfig()
model = GPT(config)

# Training is typically handled via the entry point:
# python train.py
```

## Tech Stack

* Python
* PyTorch
* Causal Transformer architecture