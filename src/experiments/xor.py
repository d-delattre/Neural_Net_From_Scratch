"""Train a small MLP on XOR and save a decision-boundary plot."""
import numpy as np
from src.core import MLP
from src.visualization.plots import plot_decision_boundary, plot_training_history


def make_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, y


def main():
    X, y = make_xor()
    model = MLP([2, 4, 1], ["tanh", "sigmoid"], "binary_cross_entropy", seed=7)
    history = model.fit(X, y, epochs=6000, learning_rate=0.5,
                        batch_size=4, shuffle=False, snapshot_every=100,
                        verbose=True)
    print("\nPredictions:")
    for x, target, prediction in zip(X, y, model.predict(X)):
        print(f"{x.astype(int)} -> {prediction[0]:.4f} (target {target[0]:.0f})")
    plot_training_history(history, "artifacts/xor_training.png")
    plot_decision_boundary(model, X, y, "artifacts/xor_decision_boundary.png")


if __name__ == "__main__":
    main()
