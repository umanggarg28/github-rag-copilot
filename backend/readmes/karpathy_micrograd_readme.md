# micrograd

A minimal autograd engine and neural network library.

## What it does

Implements a scalar-valued automatic differentiation engine to compute gradients. It provides the fundamental building blocks for neural networks, enabling the construction and training of Multi-Layer Perceptrons (MLP) using the `Value` class for backpropagation.

## Architecture

* `engine.py`: The core autograd engine centered around the `Value` class.
* `nn.py`: Neural network abstractions including `Neuron`, `Layer`, and `Module`.
* `test_engine.py`: Test suite containing `test_sanity_check` and `test_more_ops` to verify gradient correctness.

## Key Components

* `Value`: Tracks scalar values and their operations to automatically compute gradients.
* `Neuron`: Implements a single neuron that performs a weighted sum of inputs.
* `Layer`: Manages a collection of `Neuron` objects.
* `MLP`: A Multi-Layer Perceptron composed of multiple `Layer` instances.
* `Module`: A base class for all neural network components.

## Usage

```python
from micrograd.engine import Value
from micrograd.nn import MLP

# Basic autograd
a = Value(2.0)
b = Value(-3.0)
c = a * b
c.backward()
print(a.grad) # Gradient of c with respect to a

# Neural network
model = MLP(3, 4, 1) # 3 inputs, 4 hidden neurons, 1 output
```

## Tech Stack

* Python
* Scalar-based automatic differentiation design