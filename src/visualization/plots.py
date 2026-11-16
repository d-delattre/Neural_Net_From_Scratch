"""Plots driven by the actual NumPy model and its training history."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.core.activations import ACTIVATIONS


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_activation_functions(path="artifacts/activations.png"):
    x = np.linspace(-6, 6, 500)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("relu", "sigmoid", "tanh"):
        function = ACTIVATIONS[name][0]
        ax.plot(x, function(x), label=name)
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set(title="Activation functions", xlabel="pre-activation z", ylabel="activation g(z)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_training_history(history, path="artifacts/training.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"])
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    axes[1].plot(history["accuracy"])
    axes[1].set(title="Training accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, path)


def plot_decision_boundary(model, X, y, path="artifacts/decision_boundary.png"):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    pad = 0.35
    x0 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 240)
    x1 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 240)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probabilities = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(xx, yy, probabilities, levels=25, alpha=0.8)
    fig.colorbar(contour, ax=ax, label="predicted P(class=1)")
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black", s=90)
    ax.set(title="Learned XOR decision surface", xlabel="x1", ylabel="x2")
    _save(fig, path)


def plot_network_architecture(layer_sizes, path="artifacts/architecture.png"):
    """Schematic network drawing; node counts are the true layer sizes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    max_nodes = max(layer_sizes)
    positions = []
    for layer_index, count in enumerate(layer_sizes):
        ys = np.linspace(-(count - 1) / 2, (count - 1) / 2, count)
        xs = np.full(count, layer_index)
        positions.append(list(zip(xs, ys)))
    for left, right in zip(positions[:-1], positions[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                ax.plot([x1, x2], [y1, y2], color="grey", alpha=0.35, linewidth=0.8)
    for i, layer in enumerate(positions):
        for x, y in layer:
            ax.scatter(x, y, s=450, edgecolors="black", zorder=3)
        ax.text(i, -(max_nodes / 2) - 0.8, f"Layer {i}\\n{layer_sizes[i]} units",
                ha="center", va="top")
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    ax.set_ylim(-(max_nodes / 2) - 1.5, max_nodes / 2 + 0.8)
    ax.axis("off")
    ax.set_title("Fully connected MLP architecture (schematic)")
    _save(fig, path)
