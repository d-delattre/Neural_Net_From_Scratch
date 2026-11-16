"""Activation functions and their derivatives.

All functions accept and return NumPy arrays. Derivatives are evaluated
with respect to the pre-activation z, except softmax (whose full Jacobian
is intentionally not exposed because the softmax + cross-entropy gradient
has a simpler combined expression).
"""
import numpy as np


def identity(z):
    return np.asarray(z)


def identity_derivative(z):
    return np.ones_like(z, dtype=float)


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    # At z == 0 we choose the conventional subgradient 0.
    return (np.asarray(z) > 0.0).astype(float)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    # Stable evaluation avoids overflow for large negative values.
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1.0 - a)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    a = np.tanh(z)
    return 1.0 - a**2


def softmax(z):
    z = np.asarray(z, dtype=float)
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


ACTIVATIONS = {
    "identity": (identity, identity_derivative),
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "softmax": (softmax, None),
}
