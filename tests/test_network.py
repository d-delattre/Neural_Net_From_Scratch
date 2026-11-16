import numpy as np
from src.core import MLP


def test_forward_shapes():
    model = MLP([3, 5, 2], ["relu", "softmax"], "categorical_cross_entropy")
    output = model.forward(np.ones((7, 3)))
    assert output.shape == (7, 2)
    np.testing.assert_allclose(output.sum(axis=1), np.ones(7))


def test_gradient_matches_finite_difference():
    """A numerical gradient check for one weight in a tiny sigmoid/BCE model."""
    X = np.array([[0.2, -0.4], [0.7, 0.1]])
    y = np.array([[1.], [0.]])
    model = MLP([2, 3, 1], ["tanh", "sigmoid"],
                "binary_cross_entropy", seed=11)
    prediction = model.forward(X)
    model.backward(y)
    analytic = model.gradients["weights"][1][0, 0]

    original = model.weights[1][0, 0]
    eps = 1e-5
    model.weights[1][0, 0] = original + eps
    plus = model.compute_loss(y, model.forward(X))
    model.weights[1][0, 0] = original - eps
    minus = model.compute_loss(y, model.forward(X))
    model.weights[1][0, 0] = original
    numerical = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-6)


def test_training_reduces_xor_loss():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])
    model = MLP([2, 4, 1], ["tanh", "sigmoid"],
                "binary_cross_entropy", seed=7)
    history = model.fit(X, y, epochs=2500, learning_rate=0.5,
                        batch_size=4, shuffle=False)
    assert history["loss"][-1] < history["loss"][0] * 0.2
    assert model._accuracy(y, model.predict(X)) == 1.0
