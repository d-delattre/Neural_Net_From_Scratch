"""A small fully-connected MLP implemented with NumPy only."""
from __future__ import annotations

import numpy as np
from .activations import ACTIVATIONS
from .losses import LOSSES


class MLP:
    """Dense multilayer perceptron.

    Shape convention follows the project notes:
      X: (n_samples, n_features)
      W_l: (n_inputs_to_layer, n_neurons_in_layer)
      b_l: (1, n_neurons_in_layer)
      Z_l = A_(l-1) @ W_l + b_l
      A_l = activation(Z_l)

    For binary classification use a one-neuron sigmoid output and BCE.
    For multiclass classification use a softmax output and categorical CE.
    """

    def __init__(self, layer_sizes, activations, loss, seed=42):
        if len(layer_sizes) < 2:
            raise ValueError("A network needs an input and an output layer.")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Specify one activation per trainable layer.")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("Layer sizes must be positive integers.")
        if loss not in LOSSES:
            raise ValueError(f"Unknown loss: {loss}")
        if any(name not in ACTIVATIONS for name in activations):
            raise ValueError("Unknown activation in activations list.")
        if activations[-1] == "softmax" and loss != "categorical_cross_entropy":
            raise ValueError("softmax output currently requires categorical_cross_entropy.")
        if activations[-1] == "sigmoid" and loss != "binary_cross_entropy":
            raise ValueError("sigmoid output currently requires binary_cross_entropy.")

        self.layer_sizes = list(layer_sizes)
        self.activation_names = list(activations)
        self.loss_name = loss
        self.loss_function = LOSSES[loss]
        rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []
        for n_in, n_out, activation in zip(
            self.layer_sizes[:-1], self.layer_sizes[1:], self.activation_names
        ):
            # He initialization for ReLU; Xavier-style initialization otherwise.
            scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
            self.weights.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))

        self.cache = {}
        self.gradients = {}
        self.history = {"loss": [], "accuracy": [], "snapshots": []}

    def forward(self, X, return_cache=False):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array: (n_samples, n_features).")
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} features, received {X.shape[1]}."
            )

        activations = [X]
        pre_activations = []
        A = X
        for W, b, name in zip(self.weights, self.biases, self.activation_names):
            Z = A @ W + b
            activation = ACTIVATIONS[name][0]
            A = activation(Z)
            pre_activations.append(Z)
            activations.append(A)

        self.cache = {"A": activations, "Z": pre_activations}
        return (A, self.cache) if return_cache else A

    def compute_loss(self, y_true, y_pred):
        return self.loss_function(y_true, y_pred)

    def backward(self, y_true):
        """Compute gradients by applying the chain rule from output to input."""
        if not self.cache:
            raise RuntimeError("Call forward before backward.")
        y_true = np.asarray(y_true, dtype=float)
        A = self.cache["A"]
        Z = self.cache["Z"]
        y_pred = A[-1]
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} != prediction shape {y_pred.shape}")

        n = y_true.shape[0]
        # For sigmoid+BCE and softmax+categorical-CE, dL/dZ simplifies to
        # (prediction - target) / n. This is the combined chain-rule result.
        if self.loss_name == "binary_cross_entropy" and self.activation_names[-1] == "sigmoid":
            dZ = (y_pred - y_true) / n
        elif self.loss_name == "categorical_cross_entropy" and self.activation_names[-1] == "softmax":
            dZ = (y_pred - y_true) / n
        else:
            # MSE with a general differentiable output activation.
            dA = (2.0 / y_true.size) * (y_pred - y_true)
            derivative = ACTIVATIONS[self.activation_names[-1]][1]
            if derivative is None:
                raise ValueError("This loss/activation combination has no implemented derivative.")
            dZ = dA * derivative(Z[-1])

        d_weights = [None] * len(self.weights)
        d_biases = [None] * len(self.biases)

        for layer in reversed(range(len(self.weights))):
            d_weights[layer] = A[layer].T @ dZ
            d_biases[layer] = np.sum(dZ, axis=0, keepdims=True)

            if layer > 0:
                dA_previous = dZ @ self.weights[layer].T
                previous_derivative = ACTIVATIONS[self.activation_names[layer - 1]][1]
                if previous_derivative is None:
                    raise ValueError("Softmax is supported as an output activation only.")
                dZ = dA_previous * previous_derivative(Z[layer - 1])

        self.gradients = {"weights": d_weights, "biases": d_biases}
        return self.gradients

    def _apply_gradients(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.gradients["weights"][i]
            self.biases[i] -= learning_rate * self.gradients["biases"][i]

    def predict(self, X):
        return self.forward(X)

    def fit(self, X, y, epochs=1000, learning_rate=0.1, batch_size=None,
            shuffle=True, snapshot_every=25, verbose=False):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 2 or len(X) != len(y):
            raise ValueError("X and y must be 2D arrays with the same number of rows.")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive.")
        if batch_size is None:
            batch_size = len(X)
        if not 1 <= batch_size <= len(X):
            raise ValueError("batch_size must be between 1 and the dataset size.")

        rng = np.random.default_rng(123)
        self.history = {"loss": [], "accuracy": [], "snapshots": []}

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(len(X)) if shuffle else np.arange(len(X))
            for start in range(0, len(X), batch_size):
                batch_idx = indices[start:start + batch_size]
                self.forward(X[batch_idx])
                self.backward(y[batch_idx])
                self._apply_gradients(learning_rate)

            predictions = self.forward(X)
            loss_value = self.compute_loss(y, predictions)
            accuracy = self._accuracy(y, predictions)
            self.history["loss"].append(loss_value)
            self.history["accuracy"].append(accuracy)

            if epoch == 1 or epoch % snapshot_every == 0 or epoch == epochs:
                self.history["snapshots"].append({
                    "epoch": epoch,
                    "weights": [w.copy() for w in self.weights],
                    "biases": [b.copy() for b in self.biases],
                    "loss": loss_value,
                })
            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                print(f"epoch {epoch:5d} | loss {loss_value:.6f} | accuracy {accuracy:.3f}")
        return self.history

    @staticmethod
    def _accuracy(y_true, y_pred):
        if y_pred.shape[1] == 1:
            return float(np.mean((y_pred >= 0.5) == (y_true >= 0.5)))
        return float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))

    def parameter_count(self):
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))
