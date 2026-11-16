# Neural Networks from Scratch — Interactive Mathematical Laboratory

A small, inspectable multilayer perceptron (MLP) implemented with **Python and NumPy**.
The aim is to make every mathematical operation visible: affine transformations,
activation functions, loss calculation, gradients, and parameter updates.

> This is an educational implementation, not a production ML framework.

## What is implemented

- Fully connected layers using `Z = A @ W + b`
- ReLU, sigmoid, tanh, identity, and numerically stable softmax
- Binary cross-entropy, categorical cross-entropy, and mean squared error
- Backpropagation for sigmoid + binary cross-entropy and softmax + categorical cross-entropy
- Mini-batch gradient descent
- Training history and parameter snapshots for later animations
- XOR experiment and plots of activations, training history, architecture, and decision boundary
- Streamlit dashboard for basic inspection

## Mathematical convention

For a batch of `N` observations:

| Quantity | Shape |
|---|---|
| `X` | `(N, n_features)` |
| `W_l` | `(n_inputs, n_neurons)` |
| `b_l` | `(1, n_neurons)` |
| `Z_l = A_(l-1) @ W_l + b_l` | `(N, n_neurons)` |
| `A_l = g(Z_l)` | `(N, n_neurons)` |

For sigmoid output + binary cross-entropy, or softmax output + categorical
cross-entropy, the combined output-layer derivative is:

`dZ_L = (A_L - Y) / N`

Then, moving backwards through each layer:

`dW_l = A_(l-1).T @ dZ_l`

`db_l = sum(dZ_l, axis=0, keepdims=True)`

`dA_(l-1) = dZ_l @ W_l.T`

`dZ_(l-1) = dA_(l-1) * g'(Z_(l-1))`

Finally, gradient descent updates each parameter:

`W = W - learning_rate * dW`

`b = b - learning_rate * db`

The implementation follows the row-batch convention and matrix dimensions in
the project's mathematical foundations notes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,app]"
pytest
python -m src.experiments.xor
streamlit run app/dashboard.py
```

The XOR experiment writes plots to `artifacts/`.

## Repository map

```text
src/core/            NumPy model, activations, losses
src/experiments/     Reproducible experiments (currently XOR)
src/visualization/   Plots driven by real model outputs/history
app/                 Streamlit inspection interface
tests/               Unit tests and numerical gradient check
```

## Important limitations (intentional for v0.1)

- The implementation is CPU/NumPy only.
- The current `MLP.backward` supports the two standard paired output combinations
  (sigmoid + BCE, softmax + categorical cross-entropy) and MSE with supported
  elementwise output activations.
- Softmax is intended for the output layer.
- There is no regularization, optimizer abstraction, early stopping, data loader,
  or train/validation split yet.
- The Streamlit interface is a first inspection surface, not the final polished
  interactive laboratory.
- XOR is a teaching experiment, not evidence of real-world predictive performance.

## Next milestones

1. Explain and verify every line of `backward()` using a hand-worked tiny network.
2. Add a step-through forward pass and backpropagation visualizer driven by cached
   activations, pre-activations, and gradients.
3. Add gradient-descent animations and decision-boundary snapshots across epochs.
4. Add a properly split and documented real-world classification experiment.
5. Expand tests to finite-difference-check every weight and bias in multiple models.

## Design principle

**No decorative animation detached from the model.** Any future visualization must
consume the actual weights, biases, activations, gradients, and losses produced by
this implementation.
