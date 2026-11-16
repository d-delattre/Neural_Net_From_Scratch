"""Loss functions. Inputs use one row per observation."""
import numpy as np


def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) +
                          (1.0 - y_true) * np.log(1.0 - y_pred)))


def categorical_cross_entropy(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


LOSSES = {
    "binary_cross_entropy": binary_cross_entropy,
    "categorical_cross_entropy": categorical_cross_entropy,
    "mse": mse,
}
