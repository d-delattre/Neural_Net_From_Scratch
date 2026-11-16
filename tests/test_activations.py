import numpy as np
from src.core.activations import relu, sigmoid, tanh, softmax


def test_relu():
    np.testing.assert_allclose(relu(np.array([-2., 0., 3.])), [0., 0., 3.])


def test_sigmoid_midpoint_and_range():
    values = sigmoid(np.array([-1000., 0., 1000.]))
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[1], 0.5)


def test_tanh():
    np.testing.assert_allclose(tanh(np.array([0.])), [0.])


def test_softmax_is_probability_distribution():
    p = softmax(np.array([[1000., 1001., 999.], [1., 2., 3.]]))
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), [1., 1.])
